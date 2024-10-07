import argparse
from loguru import logger
import re
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from rouge_score import rouge_scorer
from utils.config import load_config
from ast import literal_eval as le
from nltk.translate import meteor
from nltk import word_tokenize
from tqdm import tqdm
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')
from src.openai_evaluator import GPT4Accuracy
from dotenv import load_dotenv
load_dotenv('.env')


def getRMSEScore(pred_values, fut_values):
    y_pred = np.reshape(pred_values, -1)
    y_true = np.reshape(fut_values, -1)
    y_pred = np.array(y_pred, dtype=np.float64)
    y_true = np.array(y_true, dtype=np.float64)

    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def getMeteorScore(outputs, pred_outputs):
    scores = [meteor([word_tokenize(output)], word_tokenize(pred_output))
              for output, pred_output in tqdm(zip(outputs, pred_outputs),
                                              total=len(outputs),
                                              desc="Calculating METEOR scores")]
    mean_score = np.mean(scores)

    return mean_score


cos_model = SentenceTransformer(
    'sentence-transformers/paraphrase-MiniLM-L6-v2')


def getCosineSimilarity(outputs, pred_outputs):
    cos_sims = [cos_sim(x, y)
                for x, y in tqdm(zip(cos_model.encode(outputs), cos_model.encode(pred_outputs)),
                                 total=len(outputs),
                                 desc="Calculating Cosine Similarities")]
    return np.mean(cos_sims)


def getROUGEScore(outputs, pred_outputs):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for output, pred_output in zip(outputs, pred_outputs):
        scores = scorer.score(output, pred_output)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    mean_rouge1 = np.mean(rouge1_scores)
    mean_rouge2 = np.mean(rouge2_scores)
    mean_rougeL = np.mean(rougeL_scores)

    return mean_rouge1, mean_rouge2, mean_rougeL


def split_forecast_by_day(text, output_window):
    days = re.split(r'(day_\d+_date:)', text)
    days = [days[i] + days[i+1] for i in range(1, len(days)-1, 2)]
    days = [day.strip() for day in days][:output_window]
    return days


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluator Configuration")
    parser.add_argument('-i', '--input_window', type=int, help='Input window size')
    parser.add_argument('-o', '--output_window', type=int, help='Output window size')
    parser.add_argument('-ms', '--mlp_stage', action='store_true',
                        help='train mlp only')
    parser.add_argument('-ls', '--llm_stage', action='store_true',
                        help='train llm only')
    parser.add_argument('-hs', '--hybrid_stage', action='store_true',
                        help='freeze mlp and train llm')
    parser.add_argument('-fs', '--final_stage', action='store_true',
                        help='train mlp and llm at the same time')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='debug by breaking loops')
    parser.add_argument('-c', '--config', type=str,
                        help='debug by breaking loops')
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = 'config/config.yaml'
    cfg = load_config(config_path)
    if args.mlp_stage:
        current_stage = cfg['mlp_dir']
    elif args.llm_stage:
        current_stage = cfg['llm_dir']
    elif args.hybrid_stage:
        current_stage = cfg['hybrid_dir']
    elif args.final_stage:
        current_stage = cfg['final_dir']

    exp_dir = f"{cfg['results_dir']}/{args.input_window}_{args.output_window}/{current_stage}"
    logger.add(f"{exp_dir}/metrics.log", rotation="1000 MB")

    batch_object_map = {}
    input_window = args.input_window
    output_window = args.output_window

    df = pd.read_csv(f"{exp_dir}/results.csv")

    df['input_times'] = df['input_times'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
    df['output_times'] = df['output_times'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
    df['input_dates'] = df['input_dates'].apply(le)
    df['output_dates'] = df['output_dates'].apply(le)
    df['input_texts'] = df['input_texts'].apply(le)
    df['output_texts'] = df['output_texts'].apply(le)
    df['pred_times'] = df['pred_times'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
    # df['pred_texts'] = df['pred_texts'].apply(le)

    # format ground truth text
    output_texts = []
    for text, date in zip(df['output_texts'], df['output_dates']):
        output_text = []
        for i in range(output_window):
            date_template = cfg['date_template'].format(index=input_window+1+i)
            text_template = cfg['text_template'].format(index=input_window+1+i)
            output_text.append(
                f"{date_template}: {date[i]}\n{text_template}: {text[i]}")
        output_texts.append('\n'.join(output_text))

    df['output_texts'] = output_texts

    pred_texts = df['pred_texts'].apply(
        lambda x: split_forecast_by_day(x, output_window)).tolist()
    output_texts = [split_forecast_by_day(
        text, output_window) for text in output_texts]
    split_data = []
    for pred, output in zip(pred_texts, output_texts):
        for p, o in zip(pred, output):
            split_data.append({'pred_text': p, 'output_text': o})

    df_evaluation = pd.DataFrame(split_data)

    rmse_score = getRMSEScore(
        np.stack(np.array(df['output_times'])), np.stack(np.array(df['pred_times'])))
    cosine_score = getCosineSimilarity(
        df_evaluation['output_text'], df_evaluation['pred_text'])
    meteor_score = getMeteorScore(
        df_evaluation['output_text'], df_evaluation['pred_text'])
    rouge_1_score, rouge_2_score, rouge_n_score = getROUGEScore(
        df_evaluation['output_text'], df_evaluation['pred_text'])

    logger.info(f"-------{exp_dir}----------")
    logger.info(f"RMSE: {round(rmse_score, 3)}")
    logger.info(f"Cosine Similarity: {round(cosine_score, 3)}")
    logger.info(f"Meteor Score: {round(meteor_score, 3)}")
    logger.info(
        f"ROUGE-1: {round(rouge_1_score, 3)}, ROUGE-2: {round(rouge_2_score, 3)}, ROUGE-N: {round(rouge_n_score, 3)}")

    # jsonl_path = f"{cfg['results_dir']}/{cfg['hybrid_dir']}/{input_window}_{output_window}/accuracy_evaluator.jsonl"
    # batch_object_id = evaluator.create_and_run_batch_job(df_evaluation, jsonl_path)
    # batch_object_map[w] = batch_object_id
