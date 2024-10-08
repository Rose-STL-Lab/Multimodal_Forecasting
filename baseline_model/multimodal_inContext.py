import concurrent.futures
import time
import os
import pandas as pd
import numpy as np
import wandb
from utils.utils import uploadToHuf, logToWandb, open_record_directory
from utils.batch_inference import batch_inference_inContext
from utils.config import load_config
from src.evaluate import getTextScore
from transformers import set_seed
from datasets import load_dataset
import torch
import multiprocessing
import argparse

def asyncRunModelChat(case, num_gpus, token, dataset, window_size, hf_dataset, split):
    results = [pd.DataFrame() for _ in range(num_gpus)]
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all[split])
    data_train = pd.DataFrame(data_all['train'])
    dataset_parts = np.array_split(data, num_gpus)
    dataset_parts = [part.reset_index(drop=True) for part in dataset_parts]
    log_path, _ = open_record_directory(dataset, f"inContext-case{case}", window_size)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
    # Create a dictionary to map each future to its corresponding index
        future_to_index = {
            executor.submit(batch_inference_inContext, dataset_parts[i], data_train, case, log_path, devices[i], token, dataset, window_size): i
            for i in range(num_gpus)
        }
        # Iterate over the completed futures
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    
    results = pd.concat(results, axis=0).reset_index(drop=True)

    return results

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="In Context Configuration")
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

    wandb.init(project="Inference-inContext",
            config={"window_size": f"{window_size}-{window_size}",
                    "dataset": dataset,
                    "model": "llama-3.1-8B-inContext"})
    hf_dataset = f"Howard881010/{dataset}-{window_size}{timestep}-inContext"
    results = asyncRunModelChat(case, num_gpus, token, dataset, window_size, hf_dataset, split, finetuned=False)
    uploadToHuf(results, hf_dataset, split, case)
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate, text_drop_count, std_error = getTextScore(
        case, split, hf_dataset, text_key_name, num_key_name, window_size
    )
    logToWandb(wandb, meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate, text_drop_count, std_error)
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
