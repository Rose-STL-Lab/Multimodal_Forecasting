from openai import OpenAI
import os
import pandas as pd
from ast import literal_eval
import json
from typing import Tuple


class OpenAIBatchAPI():
    def __init__(self,
                 jsonl_path,
                 output_path,
                 system_template="",
                 user_template="",
                 json_schema=None,
                 model_name="gpt-4o-mini",
                 description="Default OpenAIBatchAPI"
                 ):

        self.key = os.environ.get("OPENAI_APIKEY")
        self.client = OpenAI(api_key=self.key)
        self.jsonl_path = jsonl_path
        self.output_path = output_path

        self.json_schema = json_schema
        self.system_template = system_template
        self.user_template = user_template
        self.model_name = model_name
        self.description = description

    def create_batch_json(self, custom_id: str, *args: pd.DataFrame):
        response_format = None
        if self.json_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": self.json_schema
            }

        system, user = self.get_system_and_user_templates(*args)
        messages = self.message_template(system, user)
        batch_json = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "messages": messages,
                "response_format": response_format
            }
        }
        return batch_json

    def save_batch_jsons(self, batch_jsons: list[dict], jsonl_path: str) -> None:
        with open(jsonl_path, 'w') as file:
            for batch_json in batch_jsons:
                file.write(json.dumps(batch_json) + '\n')

    def start_batch_job(self, jsonl_path):
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
        return batch_object.id

    def save_batch_response(self, output_file_id, output_path: str) -> None:
        file_response = self.client.files.content(output_file_id)
        with open(output_path, "w") as file:
            file.write(file_response.text)

    def get_system_and_user_templates(self, *args) -> Tuple[str, str]:
        raise NotImplementedError

    def message_template(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return messages

    def check_status(self, batch_object_id):
        status = self.client.batches.retrieve(batch_object_id)
        return status

    def gpt_call(self, *args):
        system, user = self.get_system_and_user_templates(*args)
        messages = self.message_template(system, user)
        response_format = None
        if self.json_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": self.json_schema
            }

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format=response_format

        )
        return response.choices[0].message.content
