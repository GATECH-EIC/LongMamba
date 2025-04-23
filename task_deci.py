from mamba_ssm.utils.generation import InferenceParams
import torch
from torch.nn import CrossEntropyLoss
import os
from tqdm import tqdm
from custom_datasets.pg19 import *
from tabulate import tabulate

from utils import *


def deci_pg19(model, model_processor, model_name, merge_config):
    minimal_stride = 10
    max_amount_of_windows = merge_config['ppl_test_num_windows_per_context_len_eval']
    ce_loss = CrossEntropyLoss()
    dataset_val = get_pg19(val_only=True, model_name=model_name) 
    context_lengths = merge_config['ppl_test_context_lens_eval']
    ppl_per_context_length = []
    params_for_debug_per_example = []
    for i_ctx_len, window_size in enumerate(context_lengths):
        nlls = []
        trg_len = merge_config['ppl_test_pred_len']
        print(f'testing perplexity with context length of {window_size}, windows per sample = {max_amount_of_windows}, {trg_len} labels per window')
        for i, sample in enumerate(tqdm(dataset_val)):
            seq_len = sample['input_ids'].size(1)
            if seq_len < window_size:
                print(f'skipping sample {i}, seq_len = {seq_len//1000}K < window_size = {window_size//1000}K')
            
            stride = (seq_len-window_size)//max_amount_of_windows
            if stride < minimal_stride:
                stride = minimal_stride
            for begin_loc in range(0, seq_len-window_size, stride):
                end_loc = begin_loc + window_size
                input_ids = sample['input_ids'][:, begin_loc:end_loc].to(model.device)
                target_ids = input_ids.clone()
                with torch.no_grad():
                    model.eval()
                    if "amba" in model_name:
                        target_ids = target_ids[:, -trg_len:]
                        merge_config["resp_len"] = trg_len+1
                        inference_params = InferenceParams(max_seqlen=window_size+1, max_batch_size=input_ids.shape[0], merge_config=merge_config)
                        if "Zamba2" in model_name:
                            outputs = model(input_ids, num_logits_to_keep=100+1, merge_config=merge_config)
                        else:
                            outputs, _ = model(input_ids, num_last_tokens=trg_len+1, inference_params=inference_params)
                        logits = outputs.logits
                        neg_log_likelihood = ce_loss(logits.squeeze()[:-1], target_ids.squeeze())
                    else:
                        target_ids[:, :-trg_len] = -100  # -100 no loss sign
                        outputs = model(input_ids, labels=input_ids)
                        neg_log_likelihood = outputs.loss
                nlls.append(neg_log_likelihood)
                if end_loc == seq_len:
                    break
        ppl = torch.exp(torch.stack(nlls).mean()).cpu().to(torch.float)
        print(f'calculated perplexity: {ppl:.2f}')
        ppl_per_context_length.append(ppl)
    print(tabulate([['score:'] + [f'{x:.2f}' for x in ppl_per_context_length]], headers=['ctx len:'] + [f'{x//1000}K' for x in context_lengths] , tablefmt='pretty'))
    os.makedirs(f"pred_deci/pg19/{model_name}", exist_ok=True)
    with open(f'pred_deci/pg19/{model_name}/pg19_result.txt', 'a') as f:
        f.write(tabulate([['score:'] + [f'{x:.2f}' for x in ppl_per_context_length]], headers=['ctx len:'] + [f'{x//1000}K' for x in context_lengths] , tablefmt='pretty'))
    return None