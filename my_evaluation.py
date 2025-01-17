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
from task_deci import collate_fn_niah, deci_niah, deci_niah_debug, doc_ret, deci_pg19
from task_leval import Leval_pred, Leval_eval
from task_longbench import get_pred, scorer_e, scorer, longbench_pred
from load_model_from_config import load_model, get_decimation_config, validate_config
from load_align_target import debug

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


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def my_longbench(config, args=None, only_eval=False):
    set_seed(config['seed'])
    merge_config = validate_config(config, args)  # define chunk size 1 or False

    model_processor, model, model_name = load_model(merge_config, args)
    if not only_eval:
        longbench_pred(args, model, model_processor, model_name, merge_config)

    scores = dict()
    if merge_config['model_arch'] == "ours":
        model_name += f"_{args.align_path}-{args.c}-{args.b}-{args.our_method}"
    elif merge_config['model_arch'] == "deci":
        model_name += f"_deci-{merge_config['decimation_beta']}"
    elif merge_config['model_arch'] == "vanilla":
        model_name += "_vanilla"
    if args.long_eval_task == "e":
        path = f"pred_longbench_e/{model_name}/"
    else:
        path = f"pred_longbench/{model_name}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.long_eval_task == "e":
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    if args.long_eval_task == "e":
        out_path = f"pred_longbench_e/{model_name}/result.json"
    else:
        out_path = f"pred_longbench/{model_name}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)


def my_Leval(config, args=None, only_eval=False):
    set_seed(config['seed'])
    merge_config = validate_config(config, args)  # define chunk size 1 or False

    model_processor, model, model_name = load_model(config, args)
    if merge_config['model_arch'] == "ours":
        model_name += f"_{args.align_path}-{args.c}-{args.b}-{args.our_method}"
    elif merge_config['model_arch'] == "deci":
        model_name += f"_deci-{merge_config['decimation_beta']}"
    elif merge_config['model_arch'] == "vanilla":
        model_name += "_vanilla"

    if not only_eval:
        name_list = Leval_pred(args, model, model_processor, model_name, merge_config)
    else:
        print("only eval")
        name_list = [(f"pred_leval/{args.Leval}/{model_name}/" + name) for name in os.listdir(f"pred_leval/{args.Leval}/{model_name}") if name.endswith("jsonl")]

    all_score = dict()
    for result in name_list:
        # print("eval", result)
        args.pred_file = result
        all_score[result.split("/")[-1]] = Leval_eval(args)
    with open(f'pred_leval/{args.Leval}/{model_name}/result.json', "w") as f:
        json.dump(all_score, f, ensure_ascii=False, indent=4)


def my_deci(config, args=None):
    set_seed(config['seed'])
    merge_config = validate_config(config, args)

    # dataset settings
    path = './configs/deci_config.json'
    f = open(path)
    json_data = json.load(f)
    f.close()
    merge_config.update(json_data)
    merge_config["deci_dataset"] = args.deci_task
    # merge_config["deci_dataset"] = "none"

    model_processor, model, model_name = load_model(config, args)
    if merge_config['model_arch'] == "ours":
        model_name += f"_{args.align_path}-{args.c}-{args.b}-{args.our_method}"
    elif merge_config['model_arch'] == "deci":
        model_name += f"_deci-{merge_config['decimation_beta']}"
    elif merge_config['model_arch'] == "vanilla":
        model_name += "_vanilla"

    if args.deci_task in ["niah", "yes"]:
        sample_times = 50
        niah_map_avg = None
        for i in range(sample_times):
            niah_map = deci_niah(model, model_processor, model_name, merge_config, random_seed=123+i*2)
            if i == 0:
                niah_map_avg = niah_map / sample_times
            else:
                niah_map_avg += niah_map / sample_times
        niah_map_avg = np.round(niah_map_avg, 3)
        with open(f'pred_deci/niah/{model_name}/niah_result.txt', 'a') as f:
            f.write(f"Config: {merge_config}\n")
            f.write(tabulate(np.hstack([np.array([merge_config['niah_needle_depths_eval']]).T, niah_map_avg]), 
                            headers=['Depth / Ctx Len'] + [f'{x//1000}K' for x in merge_config['niah_context_lens_eval']], 
                            tablefmt='pretty') + "\n")
            f.write(f"Avg. acc. rate: {np.round(sum(sum(niah_map_avg))/(niah_map_avg.shape[0]*niah_map_avg.shape[1]), 4)}, Time: {datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}\n\n")
        print(tabulate(np.hstack([np.array([merge_config['niah_needle_depths_eval']]).T, niah_map_avg]), 
                        headers=['Depth / Ctx Len'] + [f'{x//1000}K' for x in merge_config['niah_context_lens_eval']], 
                        tablefmt='pretty') + "\n")
    if args.deci_task in ["pg19", "yes"]:
        deci_pg19(model, model_processor, model_name, merge_config)
    if args.deci_task in ["doc-ret", "yes"]:
        doc_ret(model, model_processor, model_name, merge_config)
    


def special_input_ppl(config, args):
    import torch.nn.functional as F
    set_seed(config['seed'])
    # args.save_para4debug = True
    merge_config = validate_config(config, args)  # define chunk size 1 or False

    tokenizer, model, model_name = load_model(config, args)

    with open(args.sample_path, 'r', encoding='utf-8') as file:
        inputs = file.read()
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
    print(f'pred_ppl/{args.align_path}-{args.our_method}_{args.model_arch}-{args.sample_path.split(".txt")[0]}.json')

    ppls = []
    perplexities = {}
    max_amount_of_windows = 20
    length = [2e3, 8e3, 16e3, 24e3, 32e3, 40e3]
    for window_size in length:
        window_size = int(window_size)
        nlls = []
        seq_len = inputs['input_ids'].size(1)
        if seq_len < window_size:
            print(f'skipping sample seq_len = {seq_len//1000}K < window_size = {window_size//1000}K')
            continue
        
        stride = (seq_len-window_size)//max_amount_of_windows
        if stride < 10:
            stride = 10

        for begin_loc in range(0, seq_len-window_size, stride):
            end_loc = begin_loc + window_size
            input_ids = inputs['input_ids'][:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()

            with torch.no_grad():
                inference_params = InferenceParams(max_seqlen=window_size+1, max_batch_size=input_ids.shape[0], merge_config=merge_config)
                target_ids = target_ids[:, -100:]
                if "Zamba2" in model_name:
                    outputs = model(input_ids, num_logits_to_keep=100+1, merge_config=merge_config)
                else:
                    outputs, _ = model(input_ids, num_last_tokens=100+1, inference_params=inference_params)
                logits = outputs.logits
                ce_loss = torch.nn.CrossEntropyLoss()
                neg_log_likelihood = ce_loss(logits.squeeze()[:-1], target_ids.squeeze())
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).mean()).cpu().to(torch.float)
        print(f'{int(window_size/1e3)}k calculated perplexity: {ppl:.2f}')
        ppls.append(ppl)
        perplexities[f'{int(window_size/1e3)}k'] = f'{ppl:.4f}'
    avg_ppl = torch.stack(ppls).mean().item()
    perplexities["average"] = f'{avg_ppl:.4f}'
    os.makedirs(f'pred_ppl/{args.sample_path.split(".txt")[0]}', exist_ok=True)
    if args.model_arch == "ours":
        with open(f'pred_ppl/{args.sample_path.split(".txt")[0]}/{model_name}_{args.align_path}-{args.our_method}-{args.c}-{args.b}.json', "w") as f:
            json.dump(perplexities, f, ensure_ascii=False, indent=4)
    elif args.model_arch == "deci":
        with open(f'pred_ppl/{args.sample_path.split(".txt")[0]}/{model_name}_{args.model_arch}-{merge_config["decimation_beta"]}.json', "w") as f:
            json.dump(perplexities, f, ensure_ascii=False, indent=4)
    else:
        with open(f'pred_ppl/{args.sample_path.split(".txt")[0]}/{model_name}_{args.model_arch}.json', "w") as f:
            json.dump(perplexities, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    
    # Each new terminal's initial command ==> export HF_HOME='/research/data/zhifan/kxia_hf'
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default='5')
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m")  # assafbk/mamba-130m-niah state-spaces/mamba-1.4b
    parser.add_argument("--model_arch", type=str, default="ours", choices=["vanilla", "ours", "deci"])

    # decay manipulation
    parser.add_argument("--align_path", type=str, default="thepileavg")  # ./artifacts/ + model_name + align_path
    parser.add_argument("--our_method", type=str, default="alpha")  # alpha bound offline dt_thre norm
    
    # for special_input_ppl()
    parser.add_argument("--perplexity", "-ppl", action="store_true")
    parser.add_argument("--sample_path", type=str, default="subseq_thepile.txt")  # subseq_lambada.txt

    # L-Eval tasks
    parser.add_argument("--Leval", "-le", type=str, default='no', choices=["no", "llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"])
    parser.add_argument("--Leval_task", type=str, default="LEval-data/Open-ended-tasks")  # None or coursera or LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks
    
    # longBench tasks
    parser.add_argument("--long_eval_task", "-lt", type=str, default='no')  # choices=["no", "yes", "e", "···"]
    
    # lm-eval tasks
    parser.add_argument("--lm_eval_task", "-lm", type=str, default='no')  # choices=["no", "···"]  # swde squadv2 fda
    parser.add_argument("--batch_size", type=int, default=64)  # for lm_eval_task

    # deciMamba tasks
    parser.add_argument("--deci_task", "-dt", type=str, default='no')    # choices=["no", "niah", "pg19", "doc-ret", "yes"]

    parser.add_argument("--b", type=float, default=1.)  # alpha factor
    parser.add_argument("--c", type=float, default=1e-30)  # channel_threshold
    parser.add_argument("--beta", type=float, default=1.0)  # deci beta

    parser.add_argument("--debug", action="store_true")  # para analysis
    parser.add_argument("--save_para4debug", "-p4d", action="store_true")  
    parser.add_argument("--save_alphaMask", "-am", action="store_true")
    parser.add_argument("--only_eval", action="store_true")  # only eval

    args = parser.parse_args()
    config = load_config(args)
    # start_datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    if args.debug:
        print("debug")
        debug(config, args)
        exit()

    if args.long_eval_task != "no":
        print("eval on long bench datasets")
        my_longbench(config, args, only_eval=args.only_eval)
    elif args.Leval != "no":
        print("eval on L-eval datasets")
        my_Leval(config, args, only_eval=args.only_eval)
    elif args.deci_task != "no":
        print("eval on deci-mamba datasets")
        my_deci(config, args)
    elif args.perplexity:
        special_input_ppl(config, args)

