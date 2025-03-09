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


def load_model(config, args):
    if "mamba" in config["base_model"]:
        wanted_dtype = torch.float32
        model_processor = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=True)

        if config['model_arch'] == 'vanilla' or 'ours' or 'deci':
            mamba_model_class = MambaLMHeadModel
        else:
            raise(f'bad mamba architecture: {config["model_arch"]}')
        
        if config['base_model'] is not None:
            print(f'loading model from checkpoint: {config["base_model"]}')
            model = mamba_model_class.from_pretrained(config['base_model'], device=f'cuda:{args.device}', dtype=wanted_dtype)
        else:
            print(f'no base_model set, loading model from checkpoint: state-spaces/mamba-1.4b')
            model = mamba_model_class.from_pretrained('state-spaces/mamba-1.4b', device=f'cuda:{args.device}', dtype=wanted_dtype)
    elif "Zamba2" in config["base_model"]:
        from zamba2.modeling_zamba2 import Zamba2ForCausalLM
        model = Zamba2ForCausalLM.from_pretrained(config['base_model'], torch_dtype=torch.bfloat16, revision="2269ae9c8a065c87dc1739ec99c9b81e5478082d").to(device=f'cuda:{args.device}')
        model_processor = AutoTokenizer.from_pretrained(config['base_model'], clean_up_tokenization_spaces=True)
    elif "opt" in config["base_model"]:
        from transformers import OPTForCausalLM, GPT2Tokenizer

        print(f'loading model from checkpoint: {config["base_model"]}')
        model = OPTForCausalLM.from_pretrained(config["base_model"], torch_dtype=torch.float32).to(f'cuda:{args.device}')
        model_processor = AutoTokenizer.from_pretrained(config["base_model"], clean_up_tokenization_spaces=True)
    else:
        print(f'loading model from checkpoint: {config["base_model"]}')
        model = AutoModelForCausalLM.from_pretrained(config["base_model"], torch_dtype=torch.float32).to(f'cuda:{args.device}')
        model_processor = AutoTokenizer.from_pretrained(config["base_model"], clean_up_tokenization_spaces=True)

    model_name = config["base_model"].split("/")[-1]
    return model_processor, model, model_name


def get_decimation_config(config):
    decimation_config = {}
    decimation_config['activate'] = True
    decimation_config['record_debug_params'] = config['save_para4debug']

    decimation_config['beta'] = config['decimation_beta']
    decimation_config['min_seq_len'] = config['decimation_min_seq_len']  # 20
    decimation_config['type'] = config['decimation_type']  # max_p
    decimation_config['L_base'] = config['decimation_max_p_L_base']  # 2000
    decimation_config['decimating_layers'] = config['decimating_layers']  # 12
    
    return decimation_config


def validate_config(config, args):
    config["model_arch"] = args.model_arch
    config["base_model"] = args.model
    if "amba" not in config["base_model"]:
        print("Using non-Mamba model, reset 'model_arch' => 'vanilla'")
        config["model_arch"] = "vanilla"
    config["save_para4debug"] = args.save_para4debug
    if args.Leval != "no":
        config["Leval"] = args.Leval
        config["Leval_task"] = args.Leval_task
    if args.long_eval_task != "no":
        config["long_eval_task"] = args.long_eval_task
    if args.lm_eval_task != "no":
        config["lm_eval_task"] = args.lm_eval_task
        config["batch_size"] = args.batch_size
    config["b"] = args.b
    config["c"] = args.c
    config["align_path"] = config["base_model"].split("/")[-1] + "-" + args.align_path
    config["our_method"] = args.our_method
    print(config)

    if config['model_arch'] == 'deci':
        path = './configs/deci_config.json'
        f = open(path)
        json_data = json.load(f)
        f.close()
        # print(json_data)
        config.update(json_data)
        if "1.4b" in args.model:
            config["decimation_max_p_L_base"] = 4000
            config["decimation_beta"] = 1
            config["decimating_layers"] = [12]
        elif "2.8b" in args.model:
            config["decimation_max_p_L_base"] = 4000
            config["decimation_beta"] = 1
            config["decimating_layers"] = [22]
        if "mamba2" in args.model:
            config["decimation_max_p_L_base"] = 4000
            config["decimation_beta"] = args.beta
            config["decimating_layers"] = [6, 20, 22, 32]

    if config['model_arch'] == 'deci' and config['deci_dataset'] == 'ppl_test':
        config['deci_num_chunks'] = 2
    
    if not config['model_arch'] == 'deci':
        config['activate_decimation'] = False

    return config