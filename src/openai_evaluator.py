"""

GPT4 Evaluagtor USAGE
=================================


from evaluator.gpt_evaluator import GPT4Evaluator
gpt4_evaluator = GPT4Evaluator()

jsonl_path = os.path.join(results_dir, "batch.jsonl")
output_path = os.path.join(results_dir, "evaluator.txt")

batch_object_id = gpt4_evaluator.create_and_run_batch_job(results, jsonl_path, 
                        output_text_column="output_text",
                        pred_text_column="pred_text")

# if ready, outputs will be returned
outputs = gpt4_evaluator.check_status_and_parse(batch_object_id, output_path)

# # if you have the results saved already, just call
# outputs = gpt4_evaluator.parse_output(output_path)

precisions, recalls, f1_scores = gpt4_evaluator.calculate_metrics(outputs)

=================================
"""

from openai import OpenAI
import os
import pandas as pd
from ast import literal_eval
import json
import numpy as np
import re


class OpeanAIBatchProcessor():
    def __init__(self, instruction: str, json_schema: dict, prompt: str, description="Multimodal Forecasting"):
        self.key = os.environ.get("OPENAI_APIKEY")
        self.client = OpenAI(api_key=self.key)
        self.instruction = instruction
        self.json_schema = json_schema
        self.prompt = prompt
        self.description = description

    def get_messages(self, target_text, pred_text) -> list[dict]:
        prompt = self.prompt.format(
            target_text=target_text, pred_text=pred_text)
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": prompt},
        ]
        return messages

    def create_and_run_batch_job(self, results: pd.DataFrame, jsonl_path: str,
                                 output_text_column="output_text",
                                 pred_text_column="pred_text") -> str:
        """
            Creates json batch from results dataframe containing ['output_text', 'pred_text']
        """

        assert ".jsonl" in jsonl_path
        batch_jsons = self.create_batch_jsons(
            results, output_text_column, pred_text_column)
        self.save_batch_json(batch_jsons, jsonl_path)

        batch_object_id = self.create_batch(jsonl_path)
        return batch_object_id

    def check_status_and_parse(self, batch_object_id, output_path: str) -> list[dict]:
        assert ".txt" in output_path
        status = self.check_status(batch_object_id)

        # check if job is done
        if status.output_file_id is None:
            print("Not ready yet")
            return status

        # save output
        self.save_batch_response(status.output_file_id, output_path)
        return self.parse_output(output_path)

    def parse_output(self, output_path: str) -> list[dict]:
        """
        Parse the output content from batch job by reading output_path.txt file
        """
        # parse output
        parsed_contents = []
        with open(output_path, 'r') as f:
            for line in f:
                output = json.loads(line)
                content = literal_eval(
                    output['response']['body']['choices'][0]['message']['content'])
                parsed_contents.append(content)
        return parsed_contents

    def create_batch_jsons(self, results: pd.DataFrame,
                           output_text_column,
                           pred_text_column) -> list[dict]:
        """
        Creates json batch from results dataframe containing ['output_text', 'pred_text']
        """
        batch_jsons = []
        for idx, row in results.iterrows():
            messages = self.get_messages(
                row[output_text_column], row[pred_text_column])

            batch_json = {
                "custom_id": f"{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                }
            }
            if self.json_schema is not None:
                batch_json['body']['response_format'] = {
                    "type": "json_schema",
                    "json_schema": self.json_schema
                }
            batch_jsons.append(batch_json)
        return batch_jsons

    def save_batch_json(self, batch_jsons: list, jsonl_path: str) -> None:
        """
        Saves batch_jsons into jsonl_path
        """
        with open(jsonl_path, 'w') as file:
            for batch_json in batch_jsons:
                file.write(json.dumps(batch_json) + '\n')

    def create_batch(self, jsonl_path: str) -> str:
        batch_input_file = self.client.files.create(
            file=open(jsonl_path, "rb"),
            purpose="batch"
        )

        batch_object = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": self.description
            }
        )
        print(f"batch job created with batch_object_id \n {batch_object.id}")
        return batch_object.id

    def check_status(self, batch_object_id):
        status = self.client.batches.retrieve(batch_object_id)
        return status

    def save_batch_response(self, output_file_id, output_path: str) -> None:
        file_response = self.client.files.content(output_file_id)
        with open(output_path, "w") as file:
            file.write(file_response.text)

    def gpt_call(self, target_text, pred_text):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.get_messages(target_text, pred_text),
            response_format={"type": "json_schema",
                             "json_schema": self.json_schema}

        )
        return response.choices[0].message.content


class GPT4SingleValue(OpeanAIBatchProcessor):

    def __init__(self, description):

        instruction = \
            """You are a helpful assistant capable of evaluating the semantic similarity between two
summaries. The semantic score you provide should be a number between 1 and 10, where
10 represents the highest level of semantic similarity (meaning the summaries convey almost
identical information), and 1 represents the lowest level of semantic similarity (meaning the
summaries convey entirely different or unrelated information). The score should reflect how
closely the meanings and key details of the two summaries align. You should only give me the
number, nothing else.
"""
        prompt = "Ground truth: {target_text} \n\n\n Output: {pred_text}"
        json_schema = None
        super().__init__(instruction, json_schema, prompt, description)

    def calculate_metrics(self, outputs):
        precisions = []
        recalls = []
        f1_scores = []
        for row in outputs:
            numbers = [float(num) for num in re.findall(r'\d+', row)]
            if len(numbers) < 3:
                continue
            tp = float(numbers[-3])
            fp = float(numbers[-2])
            fn = float(numbers[-1])
            # Since that is the result for pred output = none, we do not count it
            if tp == 0 and fp == 0 and fn == 1:
                continue

            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0

            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0

            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

    def parse_output(self, output_path: str) -> list[dict]:
        """
        Parse the output content from batch job by reading output_path.txt file
        """
        # parse output
        parsed_contents = []
        with open(output_path, 'r') as f:
            for line in f:
                output = json.loads(line)
                content = output['response']['body']['choices'][0]['message']['content']
                parsed_contents.append(content)
        return parsed_contents

class GPT4Accuracy(OpeanAIBatchProcessor):

    def __init__(self, description):

        instruction = \
            """You are a helpful assistant capable of evaluating the similarity, accuracy, and consistency of a ground truth weather forecast, and predicted weather forecast for the same day.
By comparing the semantic meaning of each sentences line by line, count the number of
1) (True Positive): Correct information presented in the prediction,
2) (False Postivie): Incorrect information presented in the prediction,
3) (False Negative): Information not presented in the prediction.
Make sure to include explanation for each step.
The end of the output should contain the counts of true positives, false positives, and false negatives.
The total count:
  - TP total count: 
  - FP total count: 
  - FN total count: 
"""
        prompt = "Ground truth weather forecast: {target_text} \n\n\n Predicted weather forecast: {pred_text}"
        json_schema = None
        super().__init__(instruction, json_schema, prompt, description)

    def calculate_metrics(self, outputs):
        precisions = []
        recalls = []
        f1_scores = []
        for row in outputs:
            numbers = [float(num) for num in re.findall(r'\d+', row)]
            if len(numbers) < 3:
                continue
            tp = float(numbers[-3])
            fp = float(numbers[-2])
            fn = float(numbers[-1])
            # Since that is the result for pred output = none, we do not count it
            if tp == 0 and fp == 0 and fn == 1:
                continue

            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0

            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0

            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

    def parse_output(self, output_path: str) -> list[dict]:
        """
        Parse the output content from batch job by reading output_path.txt file
        """
        # parse output
        parsed_contents = []
        with open(output_path, 'r') as f:
            for line in f:
                output = json.loads(line)
                content = output['response']['body']['choices'][0]['message']['content']
                parsed_contents.append(content)
        return parsed_contents
