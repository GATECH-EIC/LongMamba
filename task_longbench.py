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


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, model, tokenizer, merge_config):
    device = model.device
    merge_config["longbench_dataset"] = 'none'

    cnt = 0

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        with torch.no_grad():
            model.eval()
            if "mamba" in model_name:
                output = model.generate(
                    **input,
                    max_length=context_length+max_gen,
                    # num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    merge_config=merge_config,
                    # eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    eos_token_id=[tokenizer.eos_token_id],
                    dataset_name=dataset
                )[0]
            elif "Zamba2" in model_name:
                output = model.generate(
                    **input,
                    max_length=context_length+max_gen,
                    do_sample=False,
                    temperature=1.0,
                    merge_config=merge_config,
                    eos_token_id=[tokenizer.eos_token_id],
                )
            else:
                output = model.generate(**input, max_new_tokens=max_gen,
                                        do_sample=False, temperature=1.0, 
                                        eos_token_id=[tokenizer.eos_token_id])
        pred = tokenizer.decode(output[0][context_length:], skip_special_tokens=True)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def longbench_pred(args, model, tokenizer, model_name, merge_config):

    mp.set_start_method('spawn', force=True)
    world_size = torch.cuda.device_count()

    model2path = json.load(open("/data/kxia2/mamba/configs/model2path.json", "r"))
    model2maxlen = json.load(open("/data/kxia2/mamba/configs/model2maxlen.json", "r"))
    device = f'cuda:{args.device}'

    if "amba" in model_name or "pythia" in model_name:
        max_length = model2maxlen["mamba-1.4b"]
    elif "opt" in model_name: 
        max_length = model2maxlen["opt-125m"]
    else:
        max_length = model2maxlen[model_name]

    if args.long_eval_task == "e":
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("/data/kxia2/mamba/configs/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("/data/kxia2/mamba/configs/dataset2maxlen.json", "r"))
    
    if merge_config['model_arch'] == "ours":
        model_name += f"_{args.align_path}-{args.c}-{args.b}-{args.our_method}"
    elif merge_config['model_arch'] == "deci":
        model_name += f"_deci-{merge_config['decimation_beta']}"
    elif merge_config['model_arch'] == "vanilla":
        model_name += "_vanilla"
    for dataset in datasets:
        if args.long_eval_task == "e":
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test', trust_remote_code=True)
            if not os.path.exists(f"pred_longbench_e/{model_name}"):
                os.makedirs(f"pred_longbench_e/{model_name}")
            out_path = f"pred_longbench_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
            if not os.path.exists(f"pred_longbench/{model_name}"):
                os.makedirs(f"pred_longbench/{model_name}")
            out_path = f"pred_longbench/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        # data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        # for rank in range(world_size):
        #     p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
        #                 max_gen, prompt_format, dataset, device, model_name, model2path, out_path, model, tokenizer))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        get_pred(0, 0, data_all, max_length, max_gen, prompt_format,\
                  dataset, device, model_name, model2path, out_path, model, tokenizer, merge_config)