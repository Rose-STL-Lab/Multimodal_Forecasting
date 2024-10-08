import time
import os
import numpy as np
import wandb
from utils.utils import uploadToHuf, logToWandb
from utils.config import load_config
from src.evaluate import getTextScore
from transformers import set_seed
import torch
import multiprocessing
from baseline_model.multimodal import asyncRunModelChat
import argparse


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Zeroshot Configuration")
    parser.add_argument('-i', '--input_window', type=int,
                        help='Input window size')
    parser.add_argument('-o', '--output_window', type=int,
                        help='Output window size')
    parser.add_argument('-c', '--config', type=str,
                        help='debug by breaking loops')
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = 'config/baseline/climate.yaml'

    cfg = load_config(config_path)
    cfg = cfg['dataset']
    if args.input_window:
        cfg['input_window'] = args.input_window
    if args.output_window:
        cfg['output_window'] = args.output_window
    set_seed(cfg['seed'])
    token = os.environ.get("HF_TOKEN")
    dataset = cfg['dataset']
    split = cfg['split']
    window_size = cfg['input_window']
    timestep = cfg['timestep']
    case = cfg['case']
    text_key_name = cfg['text_key_name']
    num_key_name = cfg['num_key_name']
    num_gpus = torch.cuda.device_count()

    start_time = time.time()
    wandb.init(project="Inference-zeroshot",
            config={"window_size": f"{window_size}-{window_size}",
                    "dataset": dataset,
                    "model": "llama-3.1-8B-zeroshot"})
    hf_dataset = f"Howard881010/{dataset}-{window_size}{timestep}-zeroshot"
    results = asyncRunModelChat(case, num_gpus, token, dataset, window_size, hf_dataset, split, finetuned=False)
    uploadToHuf(results, hf_dataset, split, case)
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate, text_drop_count, std_error = getTextScore(
        case, split, hf_dataset, text_key_name, num_key_name, window_size
    )
    logToWandb(wandb, meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate, text_drop_count, std_error)
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
