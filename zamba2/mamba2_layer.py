import math
from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
from .configuration_zamba2 import Zamba2Config


try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None   

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class Mamba2Layer(nn.Module):
    def __init__(
        self,
        config: Zamba2Config,
        conv_init=None,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        # use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        #device=None,
        #dtype=None,
    ):
        factory_kwargs = {}
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.d_state = config.state_size
        self.d_conv = config.conv_dimension
        self.conv_init = conv_init
        self.expand = config.expansion_factor
        self.d_inner = (self.expand * self.d_model)

        assert self.d_inner == self.expand * self.d_model
        self.headdim = config.mamba_headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm 
        self.ngroups = config.mamba_ngroups
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = config.use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=self.config.add_bias_linear, **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=conv_dim,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=torch.bfloat16)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // self.ngroups, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.config.add_bias_linear, **factory_kwargs)

        if self.config.ft_lora:
            self.x_lora_A = nn.Linear(self.d_inner, self.config.lora_rank, bias = False)
            self.x_lora_B = nn.Linear(self.config.lora_rank, self.d_inner, bias = False)
            nn.init.zeros_(self.x_lora_B.weight)
            self.z_lora_A = nn.Linear(self.d_inner, self.config.lora_rank, bias = False)
            self.z_lora_B = nn.Linear(self.config.lora_rank, self.d_inner, bias = False)
            nn.init.zeros_(self.z_lora_B.weight)
            self.out_proj_lora_A = nn.Linear(self.d_inner, self.config.lora_rank, bias = False)
            self.out_proj_lora_B = nn.Linear(self.config.lora_rank, self.d_model, bias = False)
            nn.init.zeros_(self.out_proj_lora_B.weight)


    def forward(self, 
                u, 
                from_shared_proj=None, 
                seqlen=None, 
                seq_idx=None, 
                inference_params=None, 
                attention_mask=None,
                **kwargs,
                ):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        params_for_debug = None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.has_previous_state:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out, params_for_debug

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)        
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        input_not_masked = True
        if attention_mask is not None:
            input_not_masked = torch.all(attention_mask==1)
        if self.use_mem_eff_path and inference_params is None and input_not_masked and not self.config.ft_lora:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if attention_mask is not None and not torch.all(attention_mask==1):
                xBC = xBC * attention_mask.unsqueeze(-1)
            if conv_state is not None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            if attention_mask is not None and not torch.all(attention_mask==1):
                xBC = xBC * attention_mask.unsqueeze(-1)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            if self.config.ft_lora:
                lora_output_x = self.x_lora_A(x)
                lora_output_x = self.x_lora_B(lora_output_x)
                lora_output_z = self.z_lora_A(z)
                lora_output_z = self.z_lora_B(lora_output_z)
                x = x + lora_output_x
                z = z + lora_output_z
            
            # dt alignment
            params_for_debug = {}
            dt = F.softplus(dt + self.dt_bias)

            if inference_params.merge_config is not None and inference_params.merge_config['model_arch'] == "ours" and seqlen > 5000:
                layers_block_type = ['m', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'g', 'm', 'm']
                layer_past = layers_block_type[:(self.layer_idx+1)]
                mamba_layer_idx = len([layer for layer in layer_past if layer != 'g']) - 1

                channel_threshold = inference_params.merge_config['c']
                # whether_bound = ("bound" in inference_params.merge_config['our_method']) or ("norm" in inference_params.merge_config['our_method'])
                tA_prod_path = f"/data/kxia2/mamba/artifacts/{inference_params.merge_config['align_path']}/tA_prod/tA_prod_layer_{mamba_layer_idx}.pt"
                alpha_path = f"/data/kxia2/mamba/artifacts/{inference_params.merge_config['align_path']}/alpha/alpha_layer_{mamba_layer_idx}.pt"
                decay_path = f"/data/kxia2/mamba/artifacts/{inference_params.merge_config['align_path']}/decay/decay_layer_{mamba_layer_idx}.pt"
                dt_thre_path = f"/data/kxia2/mamba/artifacts/{inference_params.merge_config['align_path']}/delta_t-thre/delta_t-thre_layer_{mamba_layer_idx}.pt"

                tA_prod = torch.load(tA_prod_path, map_location=dt.device)
                self.channel_mask = tA_prod > channel_threshold
                whether_mask = self.channel_mask.sum() != 0

                available_values = []
                if inference_params.merge_config['our_method'] in ["alpha", "offline"]:
                    alpha_all = torch.load(alpha_path, map_location=dt.device)
                    for k in alpha_all:
                        available_values.append(int(k[:-1])*1e3)
                elif inference_params.merge_config['our_method'] in ["bound", "norm"]:  
                    decay = torch.load(decay_path, map_location=dt.device)
                elif inference_params.merge_config['our_method'] in ["dt_thre"]:
                    dt_thre_all = torch.load(dt_thre_path, map_location=dt.device)
                    for k in dt_thre_all:
                        available_values.append(int(k[:-1])*1e3)
                    
                self.slow_factor = 0
                self.topk = 2000
                dt = rearrange(dt, "b l h -> b h l")
                if inference_params.merge_config['our_method'] == "channelwise_topk" and whether_mask: 
                    topk_mask = get_topk_mask_channelwise(delta_t=dt[:, self.channel_mask], k=self.topk)
                    dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                elif inference_params.merge_config['our_method'] == "all_topk" and whether_mask:
                    topk_indice = get_top_k_token_indices(dt[:, self.channel_mask], k=self.topk)
                    topk_mask = torch.zeros_like(dt[:, self.channel_mask], dtype=torch.bool)
                    topk_mask[:,:,topk_indice] = True
                    dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)

                elif whether_mask and inference_params.merge_config['b'] != 0: 
                    # available_values = [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 10e3, 12e3, 14e3, 16e3, 20e3, 24e3, 30e3, 36e3, 44e3, 54e3, 64e3, 80e3, 96e3, 120e3, 144e3]
                    key_num = int(min(available_values, key=lambda x: abs(seqlen - x))/1e3) if available_values != [] else None
                    if "alpha" in inference_params.merge_config['our_method']:
                        channel_alpha = alpha_all[f"{key_num}k"].to(dt.device)
                        topk_mask = get_channelwise_topAlpha(delta_t=dt[:, self.channel_mask], alpha=channel_alpha[self.channel_mask]*inference_params.merge_config['b'])
                        dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                    elif "offline" in inference_params.merge_config['our_method']:
                        key_num_offline = int(min(available_values, key=lambda x: abs(seqlen - x))/1e3)
                        channel_alpha = alpha_all[f"{key_num_offline}k"].to(dt.device)
                        topk_mask, dt_thre = get_channelwise_offline(delta_t=dt[:, self.channel_mask], alpha=channel_alpha[self.channel_mask]*inference_params.merge_config['b'])
                        # self.dt_thre[f"layer_{mamba_layer_idx}"] = dt_thre
                        dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                    elif "bound" in inference_params.merge_config['our_method']:
                        topk_mask = get_channelwise_topBound(delta_t=dt[:, self.channel_mask], decay=decay[self.channel_mask]*inference_params.merge_config['b'])
                        dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                    elif "norm" in inference_params.merge_config['our_method']:
                        dt_norm = get_channelwise_normalize(delta_t=dt[:, self.channel_mask], decay=decay[self.channel_mask]*inference_params.merge_config['b'])
                        dt[:, self.channel_mask] = dt_norm.to(dt_norm.dtype)
                    elif "dt_thre" in inference_params.merge_config['our_method']:
                        channel_dt_thre_all = dt_thre_all[f"{key_num}k"].to(dt.device)
                        # self.dt_thre[f"layer_{mamba_layer_idx}"] = channel_dt_thre_all
                        topk_mask = get_channelwise_dt_threshold(delta_t=dt[:, self.channel_mask], dt_thre=channel_dt_thre_all[self.channel_mask]*inference_params.merge_config['b'])
                        dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)

                else:
                    if whether_mask:
                        print("Warning: no method is applied")
                    dt[:, self.channel_mask] = dt[:, self.channel_mask] * 0
                dt = rearrange(dt, "b h l -> b l h")

            if inference_params.merge_config['save_para4debug']:
                params_for_debug['A'] = A.clone().cpu()
                params_for_debug['Sb_x'] = B.clone().cpu()  # B before discretization
                params_for_debug['C'] = C.clone().cpu()
                params_for_debug['delta_t'] = dt.clone().cpu()  # F.softplus(dt.clone() + self.dt_bias.clone()).cpu()
                params_for_debug['B_t'] = None

            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=None,
                dt_softplus=False,
                seq_idx=seq_idx,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
            if self.config.ft_lora:
                lora_output_out = self.out_proj_lora_A(y)
                lora_output_out = self.out_proj_lora_B(lora_output_out)
                out = out + lora_output_out
        
        return out, params_for_debug

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )
        
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        if self.config.ft_lora:
            lora_output_x = self.x_lora_A(x)
            lora_output_x = self.x_lora_B(lora_output_x)
            lora_output_z = self.z_lora_A(z)
            lora_output_z = self.z_lora_B(lora_output_z)
            x = x + lora_output_x
            z = z + lora_output_z
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        if self.config.ft_lora:
            lora_output_out = self.out_proj_lora_A(y)
            lora_output_out = self.out_proj_lora_B(lora_output_out)
            out = out + lora_output_out
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=torch.bfloat16, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict_mamba:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=torch.bfloat16
            )
            inference_params.key_value_memory_dict_mamba[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict_mamba[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


'''
This function assumes that delta_t has been added with delta_bias and has gone through the softplus activation;
and then it picks the top-k tokens based on the norm of the delta_t of size (1, hidden_state, seq_len).
'''
def get_top_k_token_indices(delta_t, k=2000, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    delta_t_norm = torch.norm(delta_t, p=2, dim=1)
    delta_t_norm = delta_t_norm[:,:L_for_dec]
    k = int(min(L_for_dec, k)) # k should be less than the sequence length
    _, not_decimated = torch.topk(delta_t_norm, k, dim=1, largest=True, sorted=False)
    not_decimated, _ = torch.sort(not_decimated.squeeze())
    
    not_decimated = torch.cat([not_decimated, torch.arange(L_for_dec,L).to(not_decimated.device)])
    return not_decimated

'''
This channel selects the topk tokens for each channel (hidden_state dimension) inside delta_t (1, hidden_state, seq_len).
'''
def get_topk_mask_channelwise(delta_t, k=2000, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    k = int(min(L_for_dec, k))
    delta_t_select = delta_t[:,:,:L_for_dec]
    # using torch.quantile to get the topk threshold
    topk_threshold = torch.quantile(delta_t_select, 1 - k/L_for_dec, dim=2, keepdim=True)
    mask = delta_t_select > topk_threshold
    mask = torch.cat([mask, torch.ones_like(delta_t[:,:,L_for_dec:], dtype=torch.bool)], dim=2)
    return mask


def get_channelwise_topAlpha(delta_t, alpha=None, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    if alpha is None:
        print("Alpha is none, switch to topk method")
        return get_topk_mask_channelwise(delta_t=delta_t, k=2000, response_length=response_length)
    k_values = L_for_dec * alpha
    k_values = torch.clamp(k_values, max=L_for_dec)

    mask = torch.zeros_like(delta_t, dtype=torch.bool)

    _, sorted_indices = torch.sort(delta_t[:, :, :L_for_dec], descending=True, dim=-1)
    range_tensor = torch.arange(L_for_dec).view(1, 1, -1).expand(delta_t.size(0), delta_t.size(1), L_for_dec).to(k_values.device)
    topk_mask = range_tensor < k_values.view(delta_t.size(0), delta_t.size(1), 1)
    mask[:, :, :L_for_dec].scatter_(2, sorted_indices, topk_mask)

    if response_length > 0:
        mask[:, :, L_for_dec:] = True
    
    return mask


def get_channelwise_topBound(delta_t, decay=None, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    if decay is None:
        print("Decay bound is none, switch to topk method")
        return get_topk_mask_channelwise(delta_t=delta_t, k=2000, response_length=response_length)
    
    sorted_delta_t, sorted_indices = torch.sort(delta_t, descending=True, dim=-1)
    cumsum_delta = torch.cumsum(sorted_delta_t, dim=-1)
    cumsum_mask = cumsum_delta <= decay.view(1, -1, 1)
    topk_positions = cumsum_mask.sum(dim=-1)
    range_tensor = torch.arange(delta_t.size(-1)).view(1, 1, -1).to(topk_positions.device)

    mask = range_tensor < topk_positions.unsqueeze(-1)
    final_mask = torch.zeros_like(mask)
    final_mask.scatter_(2, sorted_indices, mask)
    
    if response_length > 0:
        final_mask[:, :, L_for_dec:] = True
    
    return final_mask


def get_channelwise_offline(delta_t, alpha=None, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length

    k_values = L_for_dec * alpha
    k_values = torch.clamp(k_values, max=L_for_dec).to(torch.int64)

    delta_t_ranked, _ = torch.sort(delta_t, descending=True, dim=-1)
    dt_thre = torch.gather(delta_t_ranked.squeeze(0), 1, k_values.unsqueeze(-1)).view(-1)

    mask = delta_t >= dt_thre.view(1, -1, 1)

    if response_length > 0:
        mask[:, :, L_for_dec:] = True
    
    return mask, dt_thre


def get_channelwise_normalize(delta_t, decay=None, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    if decay is None:
        print("Decay bound is none, switch to topk method")
        return get_topk_mask_channelwise(delta_t=delta_t, k=2000, response_length=response_length)

    delta_t_sum = torch.sum(delta_t, dim=-1, keepdim=False)
    norm = (decay.unsqueeze(0) / delta_t_sum).unsqueeze(-1)
    norm = norm.repeat(1, 1, L)
    # print(norm.shape)
    if response_length > 0:
        norm[:, :, L_for_dec:] = 1
    delta_t = delta_t*norm
    return delta_t


def get_channelwise_dt_threshold(delta_t, dt_thre=None, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    if dt_thre is None:
        print("Decay bound is none, switch to topk method")
        return get_topk_mask_channelwise(delta_t=delta_t, k=2000, response_length=response_length)

    mask = delta_t > dt_thre.unsqueeze(0).unsqueeze(-1)

    if response_length > 0:
        mask[:, :, L_for_dec:] = True

    return mask
    