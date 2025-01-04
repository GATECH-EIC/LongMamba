# import lm_eval
# from lm_eval.models.huggingface import HFLM
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


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(config):
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
        model = Zamba2ForCausalLM.from_pretrained(config['base_model'], torch_dtype=torch.bfloat16).to(device=f'cuda:{args.device}')
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


def validate_config(config):
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


def my_lm_eval(config, args=None):
    
    set_seed(config['seed'])
    merge_config = validate_config(config)  # define chunk size 1 or False
    model_processor, model, model_name = load_model(config)

    if merge_config['model_arch'] == "ours":
        model_name += f"_{args.align_path}-{args.c}-{args.b}-{args.our_method}"
    elif merge_config['model_arch'] == "deci":
        model_name += f"_deci-{merge_config['decimation_beta']}"
    elif merge_config['model_arch'] == "vanilla":
        model_name += "_vanilla"

    hflm = HFLM(pretrained=model, tokenizer=model_processor, batch_size=args.batch_size, merge_config=merge_config)
    results = lm_eval.simple_evaluate(hflm, tasks=args.lm_eval_task, batch_size=64, num_fewshot=0)
    # print(results["samples"])

    # for key, value in results["results"].items():
    #     print(f"{key}: {value}")
    #     with open('eval_custom_data_channelAlpha.txt', 'a') as f:
    #         f.write(f">>>>>>which model = [{model_name}]; factor = [{args.b}]; bound rate = [{args.c}]" + "\n")
    #         f.write(f"{key}: {value}" + "\n")
    os.makedirs(f"pred_lm_eval/{model_name}", exist_ok=True)
    with open(f'pred_lm_eval/{model_name}/result.json', 'a') as f:
        json.dump(results["results"], f, ensure_ascii=False, indent=4)


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, model, tokenizer, merge_config):
    device = model.device
    # standard_data = [args.long_eval_task]
    # lut4data = {
    #     '2wikimqa': 12,
    #     'gov_report': 4,
    #     'hotpotqa': 1,
    #     'multi_news': 3,
    #     'multifieldqa_en': 19,
    #     'multifieldqa_zh': 3,
    #     'qasper': 9,
    #     'qmsum': 1,
    #     'repobench-p': 8,
    #     'samsum': 7,
    #     'trec': 16,
    #     'triviaqa': 5,
    #     'vcsum': 3
    # }  # reverse data len<=3e3 idx

    # if dataset not in standard_data:
    #     return
    merge_config["longbench_dataset"] = 'none'

    cnt = 0

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        # if "mamba" not in model_name:
        #     if context_length > 10e3:
        #         continue
        # # else:
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
    # dist.destroy_process_group()


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


def my_longbench(config, args=None, only_eval=False):
    set_seed(config['seed'])
    merge_config = validate_config(config)  # define chunk size 1 or False

    model_processor, model, model_name = load_model(merge_config)
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

                # if "mamba" not in model_name:
                #     if prompt_length > 10e3:
                #         continue
                # # else:
                # #     if prompt_length < 3e3:
                # #         continue

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


def my_Leval(config, args=None, only_eval=False):
    set_seed(config['seed'])
    merge_config = validate_config(config)  # define chunk size 1 or False

    model_processor, model, model_name = load_model(config)
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

    _, data_loader_val = get_data_loaders_squad(config)

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
    for num_noise_docs in config['multidoc_num_noise_docs_eval']:
        cur_mean_token_count = 0
        print(f'Evaluating with {num_noise_docs} noise documents, noise injection policy: {config["multidoc_noise_injection_policy"]}')
        cur_pred_dict = {}
        cur_pred_dict['results'] = []
        noise_data_loader = DataLoader(shuffled_val_dataset, collate_fn=collate_fn_squad, batch_size=1, shuffle=False, num_workers=0).__iter__() # a bit hacky but we reset the DataLoader in every loop so we would not run out of noise documents
        for idx, batch in enumerate(tqdm(data_loader_val)):
            input_ids, prompt, golden_doc_id = get_input_ids_eval_squad(batch, model_processor, config, noise_data_loader, num_noise_docs, idx)
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
    for i_num_noise_docs, num_noise_docs in enumerate(config['multidoc_num_noise_docs_eval']):
        val_log[f'score_{num_noise_docs}_noise_docs'] = evaluator_response['scores_per_num_noise_docs'][i_num_noise_docs]
        val_log[f'mean_token_count_{num_noise_docs}_noise_docs'] = mean_token_counts[i_num_noise_docs]

    print(tabulate([['score:'] + evaluator_response["scores_per_num_noise_docs"]], headers=['num noise docs:'] + config['multidoc_num_noise_docs_eval'] , tablefmt='pretty'))

    os.makedirs(f"pred_deci/doc_ret/{model_name}", exist_ok=True)
    with open(f'pred_deci/doc_ret/{model_name}/doc_ret_result.txt', 'a') as f:
        f.write(tabulate([['score:'] + evaluator_response["scores_per_num_noise_docs"]], headers=['num noise docs:'] + config['multidoc_num_noise_docs_eval'] , tablefmt='pretty') + "\n")

    return evaluator_response

def deci_pg19(model, model_processor, model_name, merge_config):
    minimal_stride = 10
    max_amount_of_windows = config['ppl_test_num_windows_per_context_len_eval']
    ce_loss = CrossEntropyLoss()
    dataset_val = get_pg19(val_only=True) 
    context_lengths = config['ppl_test_context_lens_eval']
    ppl_per_context_length = []
    params_for_debug_per_example = []
    for i_ctx_len, window_size in enumerate(context_lengths):
        nlls = []
        trg_len = config['ppl_test_pred_len']
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

def my_deci(config, args=None):
    set_seed(config['seed'])
    merge_config = validate_config(config)

    # dataset settings
    path = './configs/deci_config.json'
    f = open(path)
    json_data = json.load(f)
    f.close()
    merge_config.update(json_data)
    merge_config["deci_dataset"] = args.deci_task
    # merge_config["deci_dataset"] = "none"

    model_processor, model, model_name = load_model(config)
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
    merge_config = validate_config(config)  # define chunk size 1 or False

    tokenizer, model, model_name = load_model(config)

    with open(args.sample_path, 'r', encoding='utf-8') as file:
        inputs = file.read()
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
    print(f'pred_ppl/{args.align_path}-{args.our_method}_{args.model_arch}-{args.sample_path.split(".txt")[0]}.json')

    ppls = []
    perplexities = {}
    max_amount_of_windows = 5
    length = [2e3, 8e3, 16e3, 24e3, 36e3, 48e3, 64e3, 80e3, 96e3]
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
            
        # plt.figure(figsize=(10, 6))
        # plt.plot(64, perplexities, marker='o', linestyle='-', color='b')
        # plt.title('Perplexity with Different Input Length')
        # plt.xlabel('Input Length(K)')
        # plt.ylabel('Perplexity')
        # plt.grid(True)
        # plt.show()


def debug(config, args):
    set_seed(config['seed'])
    merge_config = validate_config(config)  # define chunk size 1 or False

    tokenizer, model, model_name = load_model(config)

    merge_config["model_arch"] = "vanilla"
    merge_config['save_para4debug'] = True
    dataset_name = "thepile"
    with open(f'subseq_{dataset_name}.txt', 'r', encoding='utf-8') as file:
        inputS = file.read()
    inputs = tokenizer(inputS, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.size()[-1]

    for sc_, sc in enumerate(["0.16", "0.17", "0.18", "0.19", "0.20"]):
        dataset_name = f"thepile_new-clampTop{sc}-"

        samples = 5
        for idx in tqdm(range(samples)):
            os.makedirs(f'/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/decay', exist_ok=True)
            os.makedirs(f'/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/delta_t-thre', exist_ok=True)
            os.makedirs(f'/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/alpha', exist_ok=True)
            os.makedirs(f'/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/tA_prod', exist_ok=True)
            sub_input = inputs.input_ids[:, int(prompt_length/samples*idx+10):int(prompt_length/samples*idx+10+2000)]
            if "Zamba2" in model_name:
                _ = model.generate(sub_input, 
                                do_sample=False, 
                                max_length=2000 + 1, 
                                eos_token_id=[tokenizer.eos_token_id],
                                merge_config=merge_config,
                                use_cache=True)
                record = model.params_for_debug
                for key in record:
                    if key != "B_t": record[key] = torch.stack([attr for attr in record[key]])
            else:
                _, record = model.generate(sub_input, 
                                        do_sample=False, 
                                        max_length=2000 + 1, 
                                        eos_token_id=[tokenizer.eos_token_id],
                                        merge_config=merge_config,
                                        use_cache=True)
                for key in record:
                    if key != "B_t": record[key] = torch.stack([attr for attr in record[key][0]])
            # torch.save(record, f"/research/data/zhifan/kxia/artifacts/params_for_debug_{model_name}_{dataset_name}{idx:02d}.pt")
            selected_len = [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 10e3, 12e3, 14e3, 16e3, 20e3, 24e3, 30e3, 36e3, 44e3, 54e3, 64e3, 80e3, 96e3, 120e3, 144e3, 168e3,192e3]
            record['delta_t'] = rearrange(record['delta_t'], "layer b l h -> layer b h l")
            C = record['delta_t'][0][0].shape[0]
            layer_cnt = record['delta_t'].shape[0]
            for layer in tqdm(range(layer_cnt)):
                values, _ = torch.topk(record['delta_t'][layer], k=max(1, int(record['delta_t'][layer].shape[2] * (0.01 * (sc_+16)))), dim=2, largest=True, sorted=False)
                print(values.shape)
                record['delta_t'][layer] = torch.clamp(record['delta_t'][layer], max=values.min(dim=2, keepdim=True).values)
                # tA = torch.exp(torch.einsum('hs,hd->hsd', record['delta_t'][layer][0], record['A'][layer])).mean(dim=-1)
                tA = rearrange(record['delta_t'][layer], "b h l -> b l h")*record['A'][layer]
                tA = rearrange(tA, "b (c l) h -> b h c l", c=1)
                A_cumsum = torch.cumsum(tA, dim=-1)
                tA_prod = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=A_cumsum.device)[:,:,1,0]).view(-1)
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
                        # print(alpha.shape, delta_thre.shape, mod_dt[0][top_k[0].to(torch.int)] == delta_thre[0])
                torch.save(alpha_all, f"/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/alpha/alpha_layer_{layer}.pt")
                torch.save(delta_thre_all, f"/data/kxia2/mamba/artifacts/{model_name}-{dataset_name}{idx:02d}/delta_t-thre/delta_t-thre_layer_{layer}.pt")
        
        # compute avg align target
        root_path = "/data/kxia2/mamba/artifacts"
        ref_list = [f"{model_name}-{dataset_name}{d:02d}" for d in range(samples)]
        all_cnt = len(ref_list)
        print(all_cnt)

        for layer in range(layer_cnt):
            avg_alpha = {}
            avg_decay = None
            avg_tA_prod = None
            avg_delta_thre = {}
            
            for dir in ref_list:
                alpha_path = os.path.join(root_path, dir, "alpha", f"alpha_layer_{layer}.pt")
                decay_path = os.path.join(root_path, dir, "decay", f"decay_layer_{layer}.pt")
                tA_prod_path = os.path.join(root_path, dir, "tA_prod", f"tA_prod_layer_{layer}.pt")
                delta_thre_path = os.path.join(root_path, dir, "delta_t-thre", f"delta_t-thre_layer_{layer}.pt")
                alpha = torch.load(alpha_path, map_location=model.device)
                decay = torch.load(decay_path, map_location=model.device)
                tA_prod = torch.load(tA_prod_path, map_location=model.device)
                delta_thre = torch.load(delta_thre_path, map_location=model.device)

                if avg_decay is None:
                    avg_alpha = {key: value.clone()/all_cnt for key, value in alpha.items()}
                    avg_decay = decay.clone()/all_cnt
                    avg_delta_thre = {key: value.clone()/all_cnt for key, value in delta_thre.items()}
                    avg_tA_prod = tA_prod.clone()/all_cnt
                else:
                    for key in alpha:
                        avg_alpha[key] += alpha[key]/ all_cnt
                        avg_delta_thre[key] += delta_thre[key]/ all_cnt
                    avg_decay += decay/ all_cnt
                    avg_tA_prod += tA_prod/ all_cnt
            name = f"{model_name}-{dataset_name}avg"
            os.makedirs(os.path.join(root_path, name, "alpha"), exist_ok=True)
            os.makedirs(os.path.join(root_path, name, "decay"), exist_ok=True)
            os.makedirs(os.path.join(root_path, name, "tA_prod"), exist_ok=True)
            os.makedirs(os.path.join(root_path, name, "delta_t-thre"), exist_ok=True)
            torch.save(avg_alpha, os.path.join(root_path, name, "alpha", f"alpha_layer_{layer}.pt"))
            torch.save(avg_decay, os.path.join(root_path, name, "decay", f"decay_layer_{layer}.pt"))
            torch.save(avg_tA_prod, os.path.join(root_path, name, "tA_prod", f"tA_prod_layer_{layer}.pt"))
            torch.save(avg_delta_thre, os.path.join(root_path, name, "delta_t-thre", f"delta_t-thre_layer_{layer}.pt"))
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


if __name__ == '__main__':
    
    # Each new terminal's initial command ==> export HF_HOME='/research/data/zhifan/kxia_hf'
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default='5')
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m")  # assafbk/mamba-130m-niah state-spaces/mamba-1.4b
    parser.add_argument("--model_arch", type=str, default="ours", choices=["vanilla", "ours", "deci"])

    # decay manipulation
    parser.add_argument("--align_path", type=str, default="thepileavg")  # /data/kxia2/mamba/artifacts/ + model_name + align_path
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
    elif args.lm_eval_task != "no":
        print("eval on lm_eval datasets")
        my_lm_eval(config, args)
    elif args.deci_task != "no":
        print("eval on deci-mamba datasets")
        my_deci(config, args)
    elif args.perplexity:
        special_input_ppl(config, args)

