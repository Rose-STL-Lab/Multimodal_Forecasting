import torch
import pandas as pd
import torch.nn as nn
from utils.tokens import tokenize_split_days
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Union, List
from sentence_transformers import SentenceTransformer

def data_loader_to_huggingface(loader, huggingface_username):
    pass

def get_embedding_loader(df: Union[pd.DataFrame, List[str]], input_window, output_window,
                         tokenizer, llm_model, batch_size, device,
                         train_split=0.8, valid_split=0.9, column_name='temp', embd_model=None):
    n = df.shape[0]
    train_n = int(n*train_split)
    valid_n = int(n*valid_split)

    train_df = df.iloc[:train_n]
    valid_df = df.iloc[train_n:valid_n]
    test_df = df.iloc[valid_n:]

    min_value = train_df[column_name].min()
    max_value = train_df[column_name].max()

    train_dataset = TimeseriesDataset(
        train_df, input_window, output_window, tokenizer, llm_model, device, max_value, min_value, column_name=column_name, embd_model=embd_model)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=embedding_collate_fn)

    valid_dataset = TimeseriesDataset(
        valid_df, input_window, output_window, tokenizer, llm_model, device, max_value, min_value, column_name=column_name, embd_model=embd_model)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=embedding_collate_fn)

    test_dataset = TimeseriesDataset(
        test_df, input_window, output_window, tokenizer, llm_model, device, max_value, min_value, column_name=column_name, embd_model=embd_model)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=embedding_collate_fn)

    minmax_fn = train_dataset.minmax
    de_minmax_fn = train_dataset.de_minmax

    return train_loader, valid_loader, test_loader, minmax_fn, de_minmax_fn


def get_embedding_loader_from_list(df_list: Union[pd.DataFrame, List[str]], input_window, output_window,
                                   tokenizer, llm_model, batch_size, device,
                                   train_split=0.8, valid_split=0.9, column_name='Heart_Rate', embd_model=None):
    train_datasets = []
    valid_datasets = []
    test_datasets = []

    global_min_value = float('inf')
    global_max_value = float('-inf')

    for df_path in df_list:
        df = pd.read_csv(df_path)
        n = df.shape[0]
        train_n = int(n * train_split)
        valid_n = int(n * valid_split)

        train_df = df.iloc[:train_n]
        valid_df = df.iloc[train_n:valid_n]
        test_df = df.iloc[valid_n:]

        min_value = train_df[column_name].min()
        max_value = train_df[column_name].max()

        global_min_value = min(global_min_value, min_value)
        global_max_value = max(global_max_value, max_value)

        train_datasets.append(TimeseriesDataset(
            train_df, input_window, output_window, tokenizer, llm_model, device, column_name=column_name, embd_model=embd_model))
        if valid_df.shape[0] >= input_window + output_window:
            valid_datasets.append(TimeseriesDataset(
                valid_df, input_window, output_window, tokenizer, llm_model, device, column_name=column_name, embd_model=embd_model))
        if test_df.shape[0] >= input_window + output_window:
            test_datasets.append(TimeseriesDataset(
                test_df, input_window, output_window, tokenizer, llm_model, device, column_name=column_name, embd_model=embd_model))

    train_dataset = ConcatDataset(train_datasets)
    valid_dataset = ConcatDataset(valid_datasets)
    test_dataset = ConcatDataset(test_datasets)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=embedding_collate_fn)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=embedding_collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=embedding_collate_fn)

    def minmax_fn(x): return (x - global_min_value) / \
        (global_max_value - global_min_value)
    def de_minmax_fn(x): return x * (global_max_value -
                                     global_min_value) + global_min_value

    return train_loader, valid_loader, test_loader, minmax_fn, de_minmax_fn


class TimeseriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_window: int, output_window: int, tokenizer, model,
                 device, max_value=None, min_value=None, column_name='temp', embd_model=None):
        self.input_times, self.output_times, \
            self.input_texts, self.output_texts, \
            self.input_dates, self.output_dates = split_horizon(
                df, input_window, output_window, device, column_name)
        self.input_embeddings = tokenize_split_days(
            self.input_texts, tokenizer, model, device, embd_model)
        self.max_value = max_value
        self.min_value = min_value

    def __len__(self):
        return len(self.input_times)

    def __getitem__(self, idx):
        return self.input_times[idx], \
            self.output_times[idx], \
            self.input_texts[idx], \
            self.output_texts[idx], \
            self.input_dates[idx], \
            self.output_dates[idx], \
            self.input_embeddings[idx]

    def minmax(self, x):
        return (x - self.min_value) / (self.max_value - self.min_value)

    def de_minmax(self, x):
        return x * (self.max_value - self.min_value) + self.min_value


def embedding_collate_fn(data):
    input_times, output_times, input_texts, output_texts, input_dates, output_dates, input_embeddings = zip(
        *data)
    input_times = torch.stack(input_times)
    output_times = torch.stack(output_times)
    input_embeddings = torch.stack(input_embeddings)

    return input_times, output_times, input_texts, output_texts, \
        input_dates, output_dates, input_embeddings


def split_horizon(df, input_window, output_window, device, column_name='temp'):
    """
    Returns univariate timeseries
        - input_times: [N, T]
        - output_times: [N, T]
        - input_texts: [N, T, ~L]
        - output_texts: [N, T ~L]
        - input_dates: [N, T, L]
        - output_dates: [N, T, L]

    """
    total_window = input_window + output_window

    input_times = [df[column_name].iloc[i:i+input_window].tolist()
                   for i in range(len(df) - total_window + 1)]
    output_times = [df[column_name].iloc[i+input_window:i+total_window].tolist()
                    for i in range(len(df) - total_window + 1)]
    input_times = torch.tensor(input_times, device=device)
    output_times = torch.tensor(output_times, device=device)

    input_dates = [df['date'].iloc[i:i+input_window].tolist()
                   for i in range(len(df) - total_window + 1)]
    output_dates = [df['date'].iloc[i+input_window:i+total_window].tolist()
                    for i in range(len(df) - total_window + 1)]
    input_texts = [df['text'].iloc[i:i+input_window].tolist()
                   for i in range(len(df) - total_window + 1)]
    output_texts = [df['text'].iloc[i+input_window:i+total_window].tolist()
                    for i in range(len(df) - total_window + 1)]

    return input_times, output_times, input_texts, output_texts, input_dates, output_dates


def apply_chat_template(instruction_template, date_template, text_template, tokenizer,
                        input_texts, input_dates, output_dates, output_texts=None) -> list[str]:
    """
    input_texts: [b, input_window, ~N]
    input_dates: [b, input_window]
    output_dates: [b, output_window]
    output_texts: [b, output_window]
    """
    chat_inputs = []
    input_window = len(input_texts[0])
    output_window = len(output_dates[0])

    for b in range(len(input_texts)):
        chat_input = [
            {"role": "system", "content": instruction_template},
        ]
        user_content = ""
        for i in range(input_window):
            user_content += f"{date_template.format(index=i+1)}: {input_dates[b][i]}\n"
            user_content += f"{text_template.format(index=i+1)}: {input_texts[b][i]}\n"
        chat_input.append({"role": "user", "content": user_content})

        if output_texts is not None:
            assistant_content = ""
            for i in range(output_window):
                assistant_content += f"{date_template.format(index=i+1+input_window)}: {output_dates[b][i]}\n"
                assistant_content += f"{text_template.format(index=i+1+input_window)}: {output_texts[b][i]}\n"
            chat_input.append(
                {"role": "assistant", "content": assistant_content})

        chat_inputs.append(chat_input)
    prompts = tokenizer.apply_chat_template(
        chat_inputs, tokenize=False, add_generation_prompt=output_texts is None)
    return prompts
