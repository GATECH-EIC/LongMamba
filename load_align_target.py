import torch
import os
from tqdm import tqdm
from custom_datasets.pg19 import *
from einops import rearrange, repeat
import torch.nn.functional as F
from load_model_from_config import load_model, validate_config
from utils import *


def debug(config, args):
    set_seed(config['seed'])
    merge_config = validate_config(config, args) 

    tokenizer, model, model_name = load_model(config, args)

    merge_config["model_arch"] = "vanilla"
    merge_config['save_para4debug'] = True
    dataset_name = "thepile"
    with open(f'subseq_{dataset_name}.txt', 'r', encoding='utf-8') as file:
        inputS = file.read()
    inputs = tokenizer(inputS, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.size()[-1]
    print(prompt_length)

    # Ratio for clamp
    for _, sc in enumerate(["0.10", "0.15", "0.20"]):
        dataset_name = f"ablation-clampTop{sc}-"

        samples = 100
        for idx in tqdm(range(samples)):
            os.makedirs(f'./artifacts/{model_name}-{dataset_name}{idx:02d}/decay', exist_ok=True)
            os.makedirs(f'./artifacts/{model_name}-{dataset_name}{idx:02d}/delta_t-thre', exist_ok=True)
            os.makedirs(f'./artifacts/{model_name}-{dataset_name}{idx:02d}/alpha', exist_ok=True)
            os.makedirs(f'./artifacts/{model_name}-{dataset_name}{idx:02d}/tA_prod', exist_ok=True)
            os.makedirs(f'./artifacts/{model_name}-{dataset_name}{idx:02d}/A', exist_ok=True)
            
            # different pretrained length for Zamba2 and Mamba2
            if "Zamba2" in model_name:
                sub_input = inputs.input_ids[:, int(prompt_length/samples*idx+10):int(prompt_length/samples*idx+10+4000)]
                _ = model.generate(sub_input, 
                                do_sample=False, 
                                max_length=4000 + 1, 
                                eos_token_id=[tokenizer.eos_token_id],
                                merge_config=merge_config,
                                use_cache=True)
                record = model.params_for_debug
                for key in record:
                    if key != "B_t": record[key] = torch.stack([attr for attr in record[key]])
            else:
                sub_input = inputs.input_ids[:, int(prompt_length/samples*idx+10):int(prompt_length/samples*idx+10+2000)]
                _, record = model.generate(sub_input, 
                                        do_sample=False, 
                                        max_length=2000 + 1, 
                                        eos_token_id=[tokenizer.eos_token_id],
                                        merge_config=merge_config,
                                        use_cache=True)
                for key in record:
                    if key != "B_t": record[key] = torch.stack([attr for attr in record[key][0]])
            selected_len = [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 10e3, 12e3, 14e3, 16e3, 20e3, 24e3, 30e3, 36e3, 44e3, 54e3, 64e3, 80e3, 96e3, 120e3, 144e3, 168e3,192e3]
            record['delta_t'] = rearrange(record['delta_t'], "layer b l h -> layer b h l")
            C = record['delta_t'][0][0].shape[0]
            layer_cnt = record['delta_t'].shape[0]
            for layer in tqdm(range(layer_cnt)):
                values, _ = torch.topk(record['delta_t'][layer], k=max(1, int(record['delta_t'][layer].shape[2] * (float(sc)))), dim=2, largest=True, sorted=False)
                print(values.shape)
                record['delta_t'][layer] = torch.clamp(record['delta_t'][layer], max=values.min(dim=2, keepdim=True).values)
                torch.save(record['A'][layer], f"./artifacts/{model_name}-{dataset_name}{idx:02d}/A/A_layer_{layer}.pt")
                tA = rearrange(record['delta_t'][layer], "b h l -> b l h")*record['A'][layer]
                tA = rearrange(tA, "b (c l) h -> b h c l", c=1)
                A_cumsum = torch.cumsum(tA, dim=-1)
                tA_prod = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=A_cumsum.device)[:,:,1,0]).view(-1)
                torch.save(tA_prod, f"./artifacts/{model_name}-{dataset_name}{idx:02d}/tA_prod/tA_prod_layer_{layer}.pt")
                dt_sum_channels = []
                for i in range(C):
                    dt_sum = torch.sum(record['delta_t'][layer][0][i])
                    dt_sum_channels.append(dt_sum)
                torch.save(torch.stack(dt_sum_channels), f"./artifacts/{model_name}-{dataset_name}{idx:02d}/decay/decay_layer_{layer}.pt")

                alpha_all = {}
                delta_thre_all = {}
                for length in selected_len:
                    if length in [1e3, 2e3]:
                        alpha_all[f"{int(length/1e3)}k"] = torch.ones(C, device=model.device)
                        delta_thre_all[f"{int(length/1e3)}k"] = torch.zeros(C, device=model.device)
                    else:
                        standard_dt_sum = torch.stack(dt_sum_channels).to(model.device)
                        mod_dt = torch.sort(record['delta_t'][layer][0], descending=True, dim=1)[0].to(model.device)
                        mod_dt = torch.nn.functional.interpolate(mod_dt.unsqueeze(1), size=int(length), mode='linear', align_corners=False).squeeze(1)
                        mod_dt_cum = torch.cumsum(mod_dt, dim=1)

                        top_k = torch.argmax((mod_dt_cum > standard_dt_sum.unsqueeze(1)).to(torch.int), dim=1) + 1
                        alpha = top_k / length
                        alpha = torch.where(alpha <= 1, alpha, torch.tensor(1.0, device=model.device))  # Ensure alpha <= 1

                        delta_thre = torch.gather(mod_dt, 1, top_k.unsqueeze(1)).squeeze(1)

                        alpha_all[f"{int(length/1e3)}k"] = alpha
                        delta_thre_all[f"{int(length/1e3)}k"] = delta_thre
                torch.save(alpha_all, f"./artifacts/{model_name}-{dataset_name}{idx:02d}/alpha/alpha_layer_{layer}.pt")
                torch.save(delta_thre_all, f"./artifacts/{model_name}-{dataset_name}{idx:02d}/delta_t-thre/delta_t-thre_layer_{layer}.pt")
        
        # compute avg and max -based align target
        root_path = "./artifacts"
        ref_list = [f"{model_name}-{dataset_name}{d:02d}" for d in range(samples)]
        all_cnt = len(ref_list)
        print(all_cnt)

        for layer in range(layer_cnt):
            max_alpha = {}
            max_decay = None
            max_tA_prod = None
            max_A = None
            max_delta_thre = {}
            for dir in ref_list:
                # Load tensors from each directory
                alpha = torch.load(os.path.join(root_path, dir, "alpha", f"alpha_layer_{layer}.pt"), map_location=model.device)
                decay = torch.load(os.path.join(root_path, dir, "decay", f"decay_layer_{layer}.pt"), map_location=model.device)
                tA_prod = torch.load(os.path.join(root_path, dir, "tA_prod", f"tA_prod_layer_{layer}.pt"), map_location=model.device)
                A = torch.load(os.path.join(root_path, dir, "A", f"A_layer_{layer}.pt"), map_location=model.device)
                delta_thre = torch.load(os.path.join(root_path, dir, "delta_t-thre", f"delta_t-thre_layer_{layer}.pt"), map_location=model.device)
                if max_decay is None:
                    # Initialize maxima with the first set of values
                    max_alpha = {k: v.clone() for k, v in alpha.items()}
                    max_decay = decay.clone()
                    max_tA_prod = tA_prod.clone()
                    max_A = A.clone()
                    max_delta_thre = {k: v.clone() for k, v in delta_thre.items()}
                else:
                    # Update maxima for each parameter
                    for k in alpha:
                        max_alpha[k] = torch.max(max_alpha[k], alpha[k])
                        max_delta_thre[k] = torch.max(max_delta_thre[k], delta_thre[k])
                    max_decay = torch.max(max_decay, decay)
                    max_tA_prod = torch.max(max_tA_prod, tA_prod)
                    max_A = torch.max(max_A, A)
            name = f"{model_name}-{dataset_name}max"
            # Create output directories
            for sub in ("alpha", "decay", "tA_prod", "delta_t-thre", "A"):
                os.makedirs(os.path.join(root_path, name, sub), exist_ok=True)
            # Save the maximum tensors
            torch.save(max_alpha, os.path.join(root_path, name, "alpha", f"alpha_layer_{layer}.pt"))
            torch.save(max_decay, os.path.join(root_path, name, "decay", f"decay_layer_{layer}.pt"))
            torch.save(max_tA_prod, os.path.join(root_path, name, "tA_prod", f"tA_prod_layer_{layer}.pt"))
            torch.save(max_delta_thre, os.path.join(root_path, name, "delta_t-thre", f"delta_t-thre_layer_{layer}.pt"))
            torch.save(max_A, os.path.join(root_path, name, "A", f"A_layer_{layer}.pt"))

        for layer in range(layer_cnt):
            avg_alpha = {}
            avg_decay = None
            avg_tA_prod = None
            avg_A = None
            avg_delta_thre = {}
            for dir in ref_list:
                alpha_path = os.path.join(root_path, dir, "alpha", f"alpha_layer_{layer}.pt")
                decay_path = os.path.join(root_path, dir, "decay", f"decay_layer_{layer}.pt")
                tA_prod_path = os.path.join(root_path, dir, "tA_prod", f"tA_prod_layer_{layer}.pt")
                A_path = os.path.join(root_path, dir, "A", f"A_layer_{layer}.pt")
                delta_thre_path = os.path.join(root_path, dir, "delta_t-thre", f"delta_t-thre_layer_{layer}.pt")
                alpha = torch.load(alpha_path, map_location=model.device)
                decay = torch.load(decay_path, map_location=model.device)
                tA_prod = torch.load(tA_prod_path, map_location=model.device)
                A = torch.load(A_path, map_location=model.device)
                delta_thre = torch.load(delta_thre_path, map_location=model.device)
                if avg_decay is None:
                    avg_alpha = {key: value.clone()/all_cnt for key, value in alpha.items()}
                    avg_decay = decay.clone()/all_cnt
                    avg_delta_thre = {key: value.clone()/all_cnt for key, value in delta_thre.items()}
                    avg_tA_prod = tA_prod.clone()/all_cnt
                    avg_A = A.clone()/all_cnt
                else:
                    for key in alpha:
                        avg_alpha[key] += alpha[key]/ all_cnt
                        avg_delta_thre[key] += delta_thre[key]/ all_cnt
                    avg_decay += decay/ all_cnt
                    avg_tA_prod += tA_prod/ all_cnt
                    avg_A += A/ all_cnt
            name = f"{model_name}-{dataset_name}avg"
            os.makedirs(os.path.join(root_path, name, "alpha"), exist_ok=True)
            os.makedirs(os.path.join(root_path, name, "decay"), exist_ok=True)
            os.makedirs(os.path.join(root_path, name, "tA_prod"), exist_ok=True)
            os.makedirs(os.path.join(root_path, name, "delta_t-thre"), exist_ok=True)
            os.makedirs(os.path.join(root_path, name, "A"), exist_ok=True)
            torch.save(avg_alpha, os.path.join(root_path, name, "alpha", f"alpha_layer_{layer}.pt"))
            torch.save(avg_decay, os.path.join(root_path, name, "decay", f"decay_layer_{layer}.pt"))
            torch.save(avg_tA_prod, os.path.join(root_path, name, "tA_prod", f"tA_prod_layer_{layer}.pt"))
            torch.save(avg_delta_thre, os.path.join(root_path, name, "delta_t-thre", f"delta_t-thre_layer_{layer}.pt"))
            torch.save(avg_A, os.path.join(root_path, name, "A", f"A_layer_{layer}.pt"))
    exit()
    

def segsum(x, device):
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum