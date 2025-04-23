import torch
import os
import numpy as np
from tqdm import tqdm
import json
from datasets import load_dataset
from custom_datasets.pg19 import *
from utils import *

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


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, out_path, model, tokenizer, merge_config):
    device = model.device
    merge_config["longbench_dataset"] = 'none'

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        with torch.no_grad():
            model.eval()
            if "mamba" in model_name:
                output = model.generate(
                    **input,
                    max_length=context_length+max_gen,
                    do_sample=False,
                    temperature=1.0,
                    merge_config=merge_config,
                    eos_token_id=[tokenizer.eos_token_id],
                    dataset_name=dataset
                )[0]
            elif "Zamba2" in model_name:
                output = model.generate(
                    **input,
                    max_length=context_length+max_gen,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    merge_config=merge_config,
                )
            else:
                output = model.generate(**input, max_new_tokens=max_gen,
                                        eos_token_id=tokenizer.eos_token_id,
                                        do_sample=False, temperature=1.0)
        pred = tokenizer.decode(output[0][context_length:], skip_special_tokens=True)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


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
    model2maxlen = json.load(open("./configs/model2maxlen.json", "r"))
    device = f'cuda:{args.device}'

    if "mba" in model_name:
        max_length = model2maxlen["mamba-1.4b"]
    else:
        max_length = model2maxlen[model_name]

    if args.long_eval_task == "e":
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    dataset2prompt = json.load(open("./configs/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("./configs/dataset2maxlen.json", "r"))
    
    if merge_config['model_arch'] == "ours":
        model_name += f"_{args.align_path}-{args.c}-{args.b}-{args.our_method}"
    elif merge_config['model_arch'] == "deci":
        model_name += "_deci"
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
        get_pred(0, 0, data_all, max_length, max_gen, prompt_format,\
                  dataset, device, model_name, out_path, model, tokenizer, merge_config)