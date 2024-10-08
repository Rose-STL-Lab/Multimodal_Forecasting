import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import wandb
import time
from src.evaluate import getRMSEScore, getStdError, getMeteorScore, getCosineSimilarity, getROUGEScore
from utils.config import load_config
from utils.utils import split_text, logToWandb
from datasets import load_dataset
import argparse

def getTextScore(hf_dataset, text_key_name, window_size):
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all['test'])
    # number part evaluation
    pred_values = data['input_num'].to_list()
    fut_values = data['output_num'].to_list()
    rmse_loss = getRMSEScore(pred_values, fut_values)
        
    # text part evaluation
    output_texts = data['output_text'].apply(lambda x: split_text(x, text_key_name, window_size)).to_list()
    pred_texts = data['input_text'].apply(lambda x: split_text(x, text_key_name, 0)).to_list()
    output_texts = np.reshape(output_texts, -1)
    pred_texts = np.reshape(pred_texts, -1)

    meteor_score = getMeteorScore(output_texts, pred_texts)
    cosine_similarity_score = getCosineSimilarity(output_texts, pred_texts)
    rouge1, rouge2, rougeL = getROUGEScore(output_texts, pred_texts)
    std_error = getStdError(pred_values, fut_values)

    return meteor_score, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss, std_error

if __name__ == "__main__":
    # add seed
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
    token = os.environ.get("HF_TOKEN")
    dataset = cfg['dataset']
    split = cfg['split']
    window_size = cfg['input_window']
    timestep = cfg['timestep']
    case = cfg['case']
    text_key_name = cfg['text_key_name']
    num_key_name = cfg['num_key_name']

    hf_dataset = f"Howard881010/{dataset}-{window_size}{timestep}-finetuned"

    wandb.init(project="Inference-input-copy",
                config={"window_size": f"{window_size}-{window_size}",
                        "dataset": dataset,
                        "model": "input-copy"})
    
    start_time = time.time()
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, std_error = getTextScore(
        hf_dataset, text_key_name, window_size
    )
    drop_rate = 0
    text_drop_count = 0
    logToWandb(wandb, meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate, text_drop_count, std_error)
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
