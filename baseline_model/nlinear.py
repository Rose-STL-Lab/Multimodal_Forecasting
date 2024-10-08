import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from darts.models import NLinearModel
from darts import TimeSeries
import os
import pandas as pd
import numpy as np
import wandb
import time
from utils.utils import open_record_directory
from utils.config import load_config
from transformers import set_seed
from datasets import load_dataset
import ast
from src.evaluate import getRMSEScore, getStdError
import argparse

def nlinear_darts(train_input, test_input, window_size):
    # Convert to TimeSeries object required by Darts
    train_series = TimeSeries.from_values(train_input)
    model_NLinearModel = NLinearModel(input_chunk_length=window_size, output_chunk_length=window_size, pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}, )
    model_NLinearModel.fit(train_series)

    pred_value = []
    # Make predictions
    for i in range(len(test_input)):
        test_series = TimeSeries.from_values(np.array(test_input[i]))
        predictions = model_NLinearModel.predict(n=window_size, series=test_series).values().tolist()
        pred_value.append(predictions)
    
    return pred_value

def getLLMTIMEOutput(dataset, timestep, window_size, split, hf_dataset):
    data_all = load_dataset(hf_dataset)

    data = pd.DataFrame(data_all['train'])
    train_input_arr = data['input_num'].apply(lambda x: x[0]).to_list()

    data = pd.DataFrame(data_all[split])
    test_input_arr = data['input_num'].to_list()
    test_output_arr = data['output_num'].to_list()

    _, res_path = open_record_directory(
        dataset, "nlinear", window_size)

    pred_value = nlinear_darts(np.array(train_input_arr), test_input_arr, window_size)
    results = [{"pred_num": pred_value[i], "output_num": test_output_arr[i]} for i in range(len(test_input_arr))]
    results = pd.DataFrame(results, columns=['pred_num', 'output_num'])
    results.to_csv(res_path)
    return res_path

def numberEval(filename):
    data = pd.read_csv(filename)
    pred_values = data['pred_num'].apply(lambda x: ast.literal_eval(x)).to_list()
    fut_values = data['output_num'].apply(lambda x: ast.literal_eval(x)).to_list()
    rmse_loss = getRMSEScore(pred_values, fut_values)
    std_error = getStdError(pred_values, fut_values)
    return rmse_loss, std_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nlinear Configuration")
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

    hf_dataset = f"Howard881010/{dataset}-{window_size}{timestep}-finetuned"
    wandb.init(project="Inference-nlinear",
               config={"window_size": f"{window_size}-{window_size}",
                       "dataset": dataset,
                       "model": "nlinear"})
    start_time = time.time()
    
    out_filename = getLLMTIMEOutput(dataset, timestep, window_size, split, hf_dataset)
    out_rmse, std_error = numberEval(
        out_filename
    )
    wandb.log({"RMSE Scores": out_rmse})
    wandb.log({"Standard Error": std_error})
    wandb.finish()

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
