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


def Leval_pred(args, model, tokenizer, model_name, merge_config):
    device = model.device
    open_source_model = model_name
    name_list = []

    data_save_path = f"pred_leval/{args.Leval}/{open_source_model}"
    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)

    start_idx = 0
    for file_name in key_data_pairs:
        name_list.append(file_name)
        # merge_config["leval_dataset"] = file_name
        merge_config["leval_dataset"] = "none"

        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]
        header = (
            "A chat between a curious user and an artificial intelligence assistant."
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )

        sys_prompt = get_sys_prompt(args, file_name)
        for d in tqdm(data):
            document = d['input']

            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "codeU" in file_name:
                    context = document + "\n\n" + inst
                    message = sys_prompt + context
                elif "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = header + " USER: " + sys_prompt + context
                    message += " \nASSISTANT: "
                elif args.Leval == "exam_eval":
                    context = "Document is as follows. {document} \nQuestion: {inst} "
                    message = header + " USER: " + sys_prompt + context
                    message += " \nAnswer:"
                elif "coursera" in file_name:
                    context = "Document is as follows. {document} Question: {inst} "
                    message = header + " USER: " + sys_prompt + context + "\n Please only give the correct options (e.g., A)."
                    message += " \nASSISTANT: "
                else:
                    context = "Document is as follows. {document} \nInstruction: {inst} " + f"The suggested output length is around {len(out.split())} words. "
                    message = header + " USER: " + sys_prompt + context
                    message += " \nASSISTANT: "

                try:
                    text_inputs = message.format(document=document, inst=inst)
                except:
                    text_inputs = message
                save_d['prompt'] = message.replace(document, "<long input>")
                inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                prompt_length = inputs.input_ids.size()[-1]

                if "mamba" in model_name:
                    sample = model.generate(inputs.input_ids.to(model.device), 
                                            do_sample=False, 
                                            max_length=prompt_length + max_new_tokens, 
                                            eos_token_id=[tokenizer.eos_token_id],
                                            merge_config=merge_config,
                                            dataset_name=file_name,
                                            use_cache=True)[0]
                else:
                    sample = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                            do_sample=False, use_cache=True,
                                            eos_token_id=[tokenizer.eos_token_id])
                # output = tokenizer.decode(sample[prompt_length:], skip_special_tokens=True)
                output = tokenizer.decode(sample[0][prompt_length:], skip_special_tokens=True)
                save_d[f'{open_source_model}_pred'] = output
                save_d['evaluation'] = d['evaluation']

                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.", "based on the real-world knowledge and facts up until your last training") + "\nAnswer:"
                    inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                    prompt_length = inputs.input_ids.size()[-1]
                    if "mamba" in model_name:
                        sample = model.generate(**inputs, 
                                                do_sample=False, 
                                                max_length=prompt_length + max_new_tokens, 
                                                eos_token_id=[tokenizer.eos_token_id],
                                                merge_config=merge_config,
                                                dataset_name=file_name,
                                                )[0]
                    else:
                        sample = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                                do_sample=False,
                                                eos_token_id=[tokenizer.eos_token_id])
                    # output = tokenizer.decode(sample[0][prompt_length:])
                    output = tokenizer.decode(sample[0][prompt_length:], skip_special_tokens=True)
                    save_d[f'{open_source_model}_pred'] += f" [fact: {output}]"

                fw.write(json.dumps(save_d) + '\n')
        fw.close()
    return name_list


def Leval_eval(args):
    SUPPORT_METRICS = ["f1", "rouge", "exam"]

    # search for the prediction key
    pred_data = read_jsonl(args.pred_file)
    prediction_key = 0
    for key in pred_data[0]:
        if "_pred" in key:
            prediction_key = key
            break

    predictions = []
    references = []
    if  "topic_retrieval_longchat" in args.pred_file:
        references = [[], [], []]
        predictions = [[], [], []]
    elif "sci_fi" in args.pred_file:
        references = [[], []]
        predictions = [[], []]

    config_name = None

    with_options = False
    for task in with_option_tasks:
        if task in args.pred_file:
            with_options = True
            break

    for i,instance in enumerate(pred_data):
        if instance["evaluation"] not in SUPPORT_METRICS:
            continue
        if with_options:
            references.append([process_gt_mc(instance["gt"])])
            predictions.append(process_output_mc(instance[prediction_key], args.pred_file))
        elif "gsm" in args.pred_file:
            references.append([process_math(instance["gt"])])
            predictions.append(process_math(instance[prediction_key]))
        elif "codeU" in args.pred_file:
            references.append([process_gt_code(instance["gt"])])
            predictions.append(process_output_code(instance[prediction_key], instance["gt"]))
        elif "topic_retrieval_longchat" in args.pred_file:
            references[i%3].append([instance["gt"]])
            predictions[i%3].append(instance[prediction_key])
        elif "sci_fi" in args.pred_file:
            loyalty, fact = process_gt_judge(instance["gt"])
            references[0].append([loyalty])
            references[1].append([fact])
            loyalty_pred, fact_pred = process_output_judge(instance[prediction_key])
            predictions[0].append(loyalty_pred)
            predictions[1].append(fact_pred)
        else:
            references.append([instance["gt"]])
            predictions.append(instance[prediction_key])
        config_name = instance["evaluation"]
    assert config_name is not None

    if config_name in SUPPORT_METRICS:
        # print("begin evaluating:", config_name)
        LEval_metric = LEvalMetrics(config_name=config_name)
        if "topic_retrieval_longchat" in args.pred_file:
            if config_name == "exam":
                LEval_metric_trl = LEvalMetrics(config_name=config_name)
                output_str = ""
                balance_score = 0
                for i in range(len(predictions)):
                    pred = predictions[i]
                    ref = references[i]
                    metrics = LEval_metric_trl.compute(predictions=pred, references=ref)
                    output_str += f"first {i+1} sentence retrieval score: {metrics}\n"
                    balance_score += metrics["exact_match"] / len(predictions)
                # print(output_str[:-1])
                # print(f"average score of the 1st/2nd/3rd sentence retrieval: {balance_score/3}")
            else:
                metrics = LEval_metric.compute(predictions=predictions, references=references)
            print(metrics)

        elif "sci_fi" in args.pred_file:
            if config_name == "exam":
                LEval_metric_sf = LEvalMetrics(config_name=config_name)
                output_str = ""
                balance_score = 0
                for i in range(len(predictions)):
                    pred = predictions[i]
                    ref = references[i]
                    metrics = LEval_metric_sf.compute(predictions=pred, references=ref)
                    if i ==0:
                        output_str += f"loyalty score: {metrics}\n"
                    else:
                        output_str += f"fact score: {metrics}"
                    balance_score += metrics["exact_match"] / len(predictions)
                # print(output_str)
                # print(f"average score of fact and loyalty: {balance_score/2}")
            else:
                metrics = LEval_metric.compute(predictions=predictions, references=references)
            print(metrics)
        else:
            metrics = LEval_metric.compute(predictions=predictions, references=references)
            print(metrics)
        for key in metrics:
            scores = key + ": " + str(metrics[key])
            if "rouge" in key:
                scores = "rouge/rougeL" + ": " + str(metrics["rouge/rougeL"])
            break
    return scores
    # else:
    #     print(config_name, "evaluation is not ready")
    #     input("press enter to continue calculate other metrics")