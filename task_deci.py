from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams

from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
import random
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import json
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from custom_datasets.pg19 import *
import pickle
import argparse
from tabulate import tabulate
from einops import rearrange, repeat
import torch.nn.functional as F

from submodules.babilong.babilong_utils import TaskDatasetCustom, SentenceSampler, NoiseInjectionDataset

from utils import *
from LEval_config import *
from LEval_auto_eval import *

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def collate_fn_niah(data):
    batch = {}
    batch['size'] = len(data)
    batch['question'] = [elem['question'] for elem in data]
    batch['answer'] = [elem['answer'] for elem in data]
    batch['question_tokens'] = [torch.tensor([elem['question_tokens']]) for elem in data]
    batch['context_tokens'] = [torch.tensor([elem['input_tokens']]) for elem in data]
    batch['target_tokens'] = [torch.tensor([elem['target_tokens']]) for elem in data]
    batch['needle_position'] = [elem['needle_position'] for elem in data]
    return batch


def deci_niah(model, model_processor, model_name, merge_config, random_seed=123):
    test_path = "submodules/babilong/data/codes/codes_test.txt"
    noise_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    noise_sampler_test = SentenceSampler(noise_dataset['test'], tokenizer=model_processor)
    
    niah_datasets_val = []
    pct_delta = 0.1
    for needle_depth in merge_config['niah_needle_depths_eval']:
        for context_len in merge_config['niah_context_lens_eval']:
            cur_task_dataset_test = TaskDatasetCustom(test_path, max_len=1) # 1 try per depth, we init in each loop so we can get a random init of the key
            niah_datasets_val.append(NoiseInjectionDataset(task_dataset=cur_task_dataset_test,
                                noise_sampler=noise_sampler_test,
                                tokenizer=model_processor,
                                sample_size=context_len,
                                task_start_pct = max(0,needle_depth-pct_delta),
                                task_end_pct = min(1, needle_depth+pct_delta),
                                random_seed=random_seed))
    
    dataset_val = torch.utils.data.ConcatDataset(niah_datasets_val)
    data_loader_val = DataLoader(dataset_val, collate_fn=collate_fn_niah, batch_size=1, shuffle=False, num_workers=0)

    samples_df_list = []
    for idx, batch in enumerate(tqdm(data_loader_val)):
        question_tokens = batch['question_tokens'][0]
        context_tokens = batch['context_tokens'][0]
        question_post_context_tokens = model_processor(text='\nAnswer: ', return_tensors="pt").input_ids
        input_ids = torch.cat([question_tokens, context_tokens, question_post_context_tokens], dim=1).to(model.device)

        if "mamba" in model_name:
            output, _ = model.generate(input_ids, 
                                    max_length=len(input_ids[0])+10, 
                                    eos_token_id=[model_processor.eos_token_id],
                                    dataset_name="niah",
                                    merge_config=merge_config)
        else:
            output = model.generate(input_ids, max_new_tokens=10,
                                    do_sample=False, use_cache=True,
                                    eos_token_id=[model_processor.eos_token_id])
        response = model_processor.decode(output[0][len(input_ids[0]):])

        ctx_len = batch['context_tokens'][0].shape[1]
        needle_depth = batch['needle_position'][0]/ctx_len
        samples_df_list.append({'id':idx, 'response':response, 'gt':batch['answer'][0], 'ctx_len':ctx_len, 'needle_depth':f'{needle_depth:.0%}'})
    samples_df = pd.DataFrame(samples_df_list)

    responses, gts = samples_df['response'].to_list(), samples_df['gt'].to_list()
    res_flat = []
    for i in range(len(responses)):
        cur_response = responses[i].split('<|endoftext|>')[0].split(' ')
        cur_score = gts[i] in cur_response
        res_flat.append(cur_score)
    score = np.sum(res_flat)/len(res_flat)
    niah_map = np.reshape(res_flat, [len(merge_config['niah_needle_depths_eval']),len(merge_config['niah_context_lens_eval'])])
    # niah_map_str = '\n'.join('\t'.join(f'{"v" if x else "-"}' for x in y) for y in niah_map)
    
    score2str = np.vectorize(lambda x: 'v' if x else '-')
    print(tabulate(np.hstack([np.array([merge_config['niah_needle_depths_eval']]).T, score2str(niah_map)]), 
                   headers=['Depth / Ctx Len'] + [f'{x//1000}K' for x in merge_config['niah_context_lens_eval']], 
                   tablefmt='pretty'))
    
    os.makedirs(f"pred_deci/niah/{model_name}", exist_ok=True)
    # with open(f'pred_deci/niah/{model_name}/niah_result.txt', 'a') as f:
    #     f.write(f"Config: {merge_config}\n")
    #     f.write(tabulate(np.hstack([np.array([merge_config['niah_needle_depths_eval']]).T, score2str(niah_map)]), 
    #                      headers=['Depth / Ctx Len'] + [f'{x//1000}K' for x in merge_config['niah_context_lens_eval']], 
    #                      tablefmt='pretty') + "\n")
    #     f.write(f"Score: {score}, Time: {datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}\n\n")
    return niah_map

def deci_niah_debug(model, model_processor, model_name, merge_config, random_seed=123):
    merge_config["model_arch"] = "vanilla"
    merge_config['save_para4debug'] = True
    test_path = "submodules/babilong/data/codes/codes_test.txt"
    noise_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    noise_sampler_test = SentenceSampler(noise_dataset['test'], tokenizer=model_processor)
    
    niah_datasets_val = []
    pct_delta = 0.1
    merge_config['niah_needle_depths_eval'] = [i*0.01 + 0.005 for i in range(100)]
    print(merge_config['niah_needle_depths_eval'])
    for needle_depth in merge_config['niah_needle_depths_eval']:
        for context_len in merge_config['niah_context_lens_eval']:
            cur_task_dataset_test = TaskDatasetCustom(test_path, max_len=1) # 1 try per depth, we init in each loop so we can get a random init of the key
            niah_datasets_val.append(NoiseInjectionDataset(task_dataset=cur_task_dataset_test,
                                noise_sampler=noise_sampler_test,
                                tokenizer=model_processor,
                                sample_size=context_len,
                                task_start_pct = max(0,needle_depth-pct_delta),
                                task_end_pct = min(1, needle_depth+pct_delta),
                                random_seed=random_seed))
    
    dataset_val = torch.utils.data.ConcatDataset(niah_datasets_val)
    data_loader_val = DataLoader(dataset_val, collate_fn=collate_fn_niah, batch_size=1, shuffle=False, num_workers=0)

    samples_df_list = []
    for idx, batch in enumerate(tqdm(data_loader_val)):
        question_tokens = batch['question_tokens'][0]
        context_tokens = batch['context_tokens'][0]
        question_post_context_tokens = model_processor(text='\nAnswer: ', return_tensors="pt").input_ids
        input_ids = torch.cat([question_tokens, context_tokens, question_post_context_tokens], dim=1).to(model.device)

        if "mamba" in model_name:
            output, record = model.generate(input_ids, 
                                    max_length=len(input_ids[0])+10, 
                                    eos_token_id=[model_processor.eos_token_id],
                                    dataset_name="niah",
                                    merge_config=merge_config)
        else:
            output = model.generate(input_ids, max_new_tokens=10,
                                    do_sample=False, use_cache=True,
                                    eos_token_id=[model_processor.eos_token_id])
        dataset_name = "niah"    
        os.makedirs(f'/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/decay', exist_ok=True)
        os.makedirs(f'/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/delta_t-thre', exist_ok=True)
        os.makedirs(f'/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/alpha', exist_ok=True)
        os.makedirs(f'/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/tA_prod', exist_ok=True)
        
        for key in record:
            if key != "B_t": record[key] = torch.stack([attr for attr in record[key][0]])
        torch.save(record, f"/research/data/zhifan/kxia/artifacts/params_for_debug_{model_name}_{dataset_name}{idx:02d}.pt")
        selected_len = [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 10e3, 12e3, 14e3, 16e3, 20e3, 24e3, 30e3, 36e3, 44e3, 54e3, 64e3, 80e3, 96e3, 120e3, 144e3, 168e3,192e3]
        C = record['delta_t'][0][0].shape[0]
        layer_cnt = record['delta_t'].shape[0]
        for layer in tqdm(range(layer_cnt)):
            tA = torch.exp(torch.einsum('hs,hd->hsd', record['delta_t'][layer][0], record['A'][layer])).mean(dim=-1)
            tA_prod = torch.prod(tA, dim=-1)
            torch.save(tA_prod, f"/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/tA_prod/tA_prod_layer_{layer}.pt")
            dt_sum_channels = []
            for i in range(C):
                dt_sum = torch.sum(record['delta_t'][layer][0][i])
                dt_sum_channels.append(dt_sum)
            # dt_sum_all.append(dt_sum_channels)
            torch.save(torch.stack(dt_sum_channels), f"/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/decay/decay_layer_{layer}.pt")

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
                    mod_dt_cum = torch.cumsum(mod_dt, dim=1)  # Shape [C, length]

                    top_k = torch.argmax((mod_dt_cum > standard_dt_sum.unsqueeze(1)).to(torch.int), dim=1) + 1  # Shape [C]
                    alpha = top_k / length
                    alpha = torch.where(alpha <= 1, alpha, torch.tensor(1.0, device=model.device))  # Ensure alpha <= 1

                    delta_thre = torch.gather(mod_dt, 1, top_k.unsqueeze(1)).squeeze(1)

                    alpha_all[f"{int(length/1e3)}k"] = alpha
                    delta_thre_all[f"{int(length/1e3)}k"] = delta_thre
            torch.save(alpha_all, f"/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/alpha/alpha_layer_{layer}.pt")
            torch.save(delta_thre_all, f"/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/delta_t-thre/delta_t-thre_layer_{layer}.pt")
    exit()

def doc_ret(model, model_processor, model_name, merge_config):
    
    def collate_fn_squad(data):
        batch = {}
        batch['size'] = len(data)
        batch['inputs'] = [f'{elem["question"]}\n\n{elem["context"]}' for elem in data]
        batch['outputs'] = [elem['answers']['text'] for elem in data]
        batch['ids'] = [elem['id'] for elem in data]
        batch['titles'] = [elem['title'] for elem in data]
        batch['questions'] = [elem["question"] for elem in data]
        batch['contexts'] = [elem["context"] for elem in data]
        return batch

    def run_squad_retrieve_evaluator(pred_dicts):
        scores_per_num_noise_docs = []
        for i, pred_dict in enumerate(pred_dicts):
            cur_score = 0
            for pred in pred_dict['results']:
                if pred['pred'] == pred['gt']:
                    cur_score += 1
            scores_per_num_noise_docs.append(cur_score/len(pred_dict['results']))

        return {'score': np.mean(scores_per_num_noise_docs), 'scores_per_num_noise_docs': scores_per_num_noise_docs}

    def get_data_loaders_squad(config, final_eval_mode=False):
        with open('./submodules/multidoc_squad/has_answer_indices_val.pkl', 'rb') as f:
            validation_indices_list = pickle.load(f)

        USER_AGENT = get_datasets_user_agent()
        dataset = load_dataset("rajpurkar/squad_v2")
        dataset['validation'] = dataset['validation'].select(validation_indices_list)

        if final_eval_mode:
            data_loader_val = DataLoader(dataset['validation'], collate_fn=collate_fn_squad, batch_size=1, shuffle=False, num_workers=0)
            return data_loader_val
        
        else:
            dataset_smaller_val_split = dataset['validation'].train_test_split(test_size=100/dataset['validation'].num_rows, seed=111) # we set the same seed because we do want to shuffle the dataset before selecting the val set, but want it to be consistent every time.
            
            data_loader_val = DataLoader(dataset_smaller_val_split['test'], collate_fn=collate_fn_squad, batch_size=1, shuffle=False, num_workers=0)
            return None, data_loader_val

    _, data_loader_val = get_data_loaders_squad(merge_config)

    def inject_noise_to_context(config, golden_doc, noise_data_loader, idx, num_noise_docs, query, is_eval=False):
        
        # get golden doc location
        if config['multidoc_noise_injection_policy'] == 'random_loc':
            if is_eval:
                with open('./submodules/multidoc_squad/random_indices_for_num_docs_before_golden_val.pkl', 'rb') as f:
                    noise_dataset_shuffled_indices = pickle.load(f)
            else:
                with open('./submodules/multidoc_squad/random_indices_for_num_docs_before_golden_train.pkl', 'rb') as f:
                    noise_dataset_shuffled_indices = pickle.load(f)
            num_docs_before_golden = noise_dataset_shuffled_indices[idx] % (num_noise_docs+1)
        else:
            num_docs_before_golden = 0
        
        # sample noise and inject doc
        noise_docs_before_golden = []
        for s in range(num_docs_before_golden):
            noise_docs_before_golden.append(noise_data_loader.__next__()['contexts'][0])
        
        noise_docs_after_golden = []
        for s in range(num_noise_docs-num_docs_before_golden):
            noise_docs_after_golden.append(noise_data_loader.__next__()['contexts'][0])
        
        all_docs = noise_docs_before_golden + [golden_doc] + noise_docs_after_golden

        noisy_context = ''
        doc_ids = random.sample(range(0, 1000), num_noise_docs+1)
        for i_doc, doc in enumerate(all_docs):
            noisy_context += f' <|Query|> {query} <|Document {doc_ids[i_doc]}|> {doc}'
        
        return noisy_context, doc_ids[i_doc]

    def get_input_ids_eval_squad(batch, model_processor, config, noise_data_loader, num_noise_docs, i):
        prompt, golden_doc_id = inject_noise_to_context(config, batch["contexts"][0], noise_data_loader, i, num_noise_docs, batch["questions"][0], is_eval=True)
        prompt = prompt + ' <|Answer|> <|Document '
        input_ids = model_processor(text=prompt, return_tensors="pt").input_ids.to(f'cuda:{args.device}')
        return input_ids, prompt, golden_doc_id
    
    def update_results_eval(pred_dict, samples_df_list, batch, idx, response, prompt, squad_num_noise_docs=None):
        pred_dict['results'].append({'id': batch["ids"][0], 'pred': response, 'gt': batch['outputs'][0]})
        samples_df_list.append({'id':batch["ids"][0], 'num_noise_docs':squad_num_noise_docs, 'prompt':prompt[0:200], 'response':response, 'gt':batch['outputs'][0]})
        
        return pred_dict, samples_df_list

    with open('./submodules/multidoc_squad/noise_docs_indices_val.pkl', 'rb') as f:
        noise_dataset_shuffled_indices_val = pickle.load(f)
    dataset_val = load_dataset("rajpurkar/squad_v2", split="validation")
    shuffled_val_dataset = dataset_val.select(noise_dataset_shuffled_indices_val.tolist()*10)
    
    samples_df_list = []
    val_log = {}
    pred_dicts = []
    mean_token_counts = []
    for num_noise_docs in merge_config['multidoc_num_noise_docs_eval']:
        cur_mean_token_count = 0
        print(f'Evaluating with {num_noise_docs} noise documents, noise injection policy: {merge_config["multidoc_noise_injection_policy"]}')
        cur_pred_dict = {}
        cur_pred_dict['results'] = []
        noise_data_loader = DataLoader(shuffled_val_dataset, collate_fn=collate_fn_squad, batch_size=1, shuffle=False, num_workers=0).__iter__() # a bit hacky but we reset the DataLoader in every loop so we would not run out of noise documents
        for idx, batch in enumerate(tqdm(data_loader_val)):
            input_ids, prompt, golden_doc_id = get_input_ids_eval_squad(batch, model_processor, merge_config, noise_data_loader, num_noise_docs, idx)
            # print(input_ids.shape)
            # with open(f"doc_ret_sample.txt", "w") as f:
            #     f.write(prompt)
            #     exit()
            batch['outputs'][0] = f'{golden_doc_id}|>'
            # output, params_for_debug = model.generate(input_ids, max_length=len(input_ids[0])+10, eos_token_id=model_processor.eos_token_id)
            if "mamba" in model_name:
                output, _ = model.generate(input_ids, 
                                        max_length=len(input_ids[0])+10, 
                                        eos_token_id=[model_processor.eos_token_id],
                                        dataset_name="squad",
                                        merge_config=merge_config)
            else:
                output = model.generate(input_ids, max_new_tokens=10,
                                        do_sample=False, use_cache=True,
                                        eos_token_id=[model_processor.eos_token_id])
            
            response = model_processor.decode(output[0][len(input_ids[0]):])
            response = response.split('|>')[0] + '|>'
            cur_pred_dict, samples_df_list = update_results_eval(cur_pred_dict, samples_df_list, batch, idx, response, prompt, num_noise_docs)
            cur_mean_token_count += input_ids.shape[1]

        pred_dicts.append(cur_pred_dict)
        mean_token_counts.append(cur_mean_token_count/len(data_loader_val))

    evaluator_response = run_squad_retrieve_evaluator(pred_dicts)
    val_log['score'] = evaluator_response['score']
    for i_num_noise_docs, num_noise_docs in enumerate(merge_config['multidoc_num_noise_docs_eval']):
        val_log[f'score_{num_noise_docs}_noise_docs'] = evaluator_response['scores_per_num_noise_docs'][i_num_noise_docs]
        val_log[f'mean_token_count_{num_noise_docs}_noise_docs'] = mean_token_counts[i_num_noise_docs]

    print(tabulate([['score:'] + evaluator_response["scores_per_num_noise_docs"]], headers=['num noise docs:'] + merge_config['multidoc_num_noise_docs_eval'] , tablefmt='pretty'))

    os.makedirs(f"pred_deci/doc_ret/{model_name}", exist_ok=True)
    with open(f'pred_deci/doc_ret/{model_name}/doc_ret_result.txt', 'a') as f:
        f.write(tabulate([['score:'] + evaluator_response["scores_per_num_noise_docs"]], headers=['num noise docs:'] + merge_config['multidoc_num_noise_docs_eval'] , tablefmt='pretty') + "\n")

    return evaluator_response

def deci_pg19(model, model_processor, model_name, merge_config):
    minimal_stride = 10
    max_amount_of_windows = merge_config['ppl_test_num_windows_per_context_len_eval']
    ce_loss = CrossEntropyLoss()
    dataset_val = get_pg19(val_only=True) 
    context_lengths = merge_config['ppl_test_context_lens_eval']
    ppl_per_context_length = []
    params_for_debug_per_example = []
    for i_ctx_len, window_size in enumerate(context_lengths):
        nlls = []
        trg_len = merge_config['ppl_test_pred_len']
        print(f'testing perplexity with context length of {window_size}, windows per sample = {max_amount_of_windows}, {trg_len} labels per window')
        for i, sample in enumerate(tqdm(dataset_val)):
            seq_len = sample['input_ids'].size(1)
            tokenizer_neox = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=True)
            text = tokenizer_neox.decode(sample['input_ids'][0], skip_special_tokens=True)
            sample['input_ids'] = model_processor(text=text, return_tensors="pt").input_ids.to(f'cuda:{args.device}')
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