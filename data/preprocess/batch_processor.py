from preprocess.opeani_batch_processor import OpenAIBatchAPI
import pandas as pd
from typing import Tuple
from ast import literal_eval
import json
from collections import defaultdict
from tqdm import tqdm
from PROMPTS import (CLIMATE_CHUNK_INSTRUCTION, CLIMATE_REFINER_INSTRUCTION, CLIMATE_REFINER_TEMPLATE,
                     MEDICAL_CHUNK_INSTRUCTION, MEDICAL_REFINER_INSTRUCTION, MEDICAL_REFINER_TEMPLATE)


class ClimateChunkProcessor(OpenAIBatchAPI):
    instruction = CLIMATE_CHUNK_INSTRUCTION
    system_template = ""
    user_template = ""

    chunk_length = 1000
    model_name = "gpt-4o-mini"
    json_schema = None

    # ----- example for json schema -----
    # sub_json = {"type": "string"}
    # json_schema = {
    #     "name": "evaluate_climate_report",
    #     "strict": True,
    #     "schema": {
    #         "type": "object",
    #         "properties": {
    #             "temperature_trend": sub_json,
    #             "precipitation_trend": sub_json,
    #             "humidity_trend": sub_json,
    #             "wind_speed_trend": sub_json,
    #             "overall_summary": {
    #                 "type": "string"
    #             }
    #         },
    #         "additionalProperties": False,
    #         "required": ["temperature_trend", "precipitation_trend", "humidity_trend", "wind_speed_trend", "overall_summary"]
    #     }
    # }

    def __init__(self,
                 jsonl_path,
                 output_path,
                 description="Climate Chunk Processing"):

        super().__init__(jsonl_path, output_path, self.system_template,
                         self.user_template, self.json_schema, self.model_name, description)

    def get_system_and_user_templates(self, chunk) -> Tuple[str, str]:
        system = self.instruction
        user = chunk
        return system, user

    def create_batch(self, path):
        batch_jsons = []
        df = pd.read_csv(path)
        for row_idx, row in df.iterrows():
            document = row['Text']
            for chunk_idx, chunk_start in enumerate(range(0, len(document), self.chunk_length)):
                json_batch = self.create_batch_json(
                    f"{row_idx}_{chunk_idx}", row['Text'][chunk_start: chunk_start+self.chunk_length])
                batch_jsons.append(json_batch)

        self.save_batch_jsons(batch_jsons, self.jsonl_path)
        batch_object_id = self.start_batch_job(self.jsonl_path)
        return batch_object_id

    def parse_output(self, output_path: str) -> list[dict]:
        """
        Parse the output content from batch job by reading output_path.txt file
        """
        parsed_contents = []
        with open(output_path, 'r') as f:
            for line in f:
                output = json.loads(line)
                content = output['response']['body']['choices'][0]['message']['content']
                parsed_contents.append(content)
        return parsed_contents


class ClimateRefiner(OpenAIBatchAPI):
    instruction = CLIMATE_REFINER_INSTRUCTION
    system_template = ""
    user_template = CLIMATE_REFINER_TEMPLATE
    chunk_length = 1000
    model_name = "gpt-4o-mini"
    json_schema = None

    def __init__(self,
                 jsonl_path,
                 output_path,
                 description="Climate Refiner"):

        super().__init__(jsonl_path, output_path, self.system_template,
                         self.user_template, self.json_schema, self.model_name, description)

    def get_system_and_user_templates(self, input, summary) -> Tuple[str, str]:
        system = self.instruction
        user = self.user_template.format(input=input, summary=summary)
        return system, user

    def parse_output(self, output_path: str) -> list[dict]:
        parsed_contents = defaultdict(list)  # {row_idx, chunk_idx}
        errors = []
        with open(output_path, 'r') as f:
            for line in f:
                output = json.loads(line)
                custom_id = output['custom_id']
                row_idx = custom_id
                chunk_idx = custom_id
                # row_idx, chunk_idx = custom_id.split('_')
                # row_idx = int(row_idx)
                # chunk_idx = int(chunk_idx)
                try:
                    content = literal_eval(
                        output['response']['body']['choices'][0]['message']['content'])
                    parsed_contents[row_idx].append((chunk_idx, content))
                except:
                    errors.append((row_idx, chunk_idx))

        formatted_contents = {}
        for row_idx, data in tqdm(parsed_contents.items()):
            json_data = []
            # ensure it is sorted.
            data = sorted(data, key=lambda x: x[0])
            for chunk_idx, json_summary in data:
                # for chunk_idx, text in json_summary.items():
                text = json_summary['overall_summary']
                if text != '':
                    json_data.append(text.capitalize())
            formatted_contents[row_idx] = json_data
        sorted_contents = sorted(formatted_contents.items())
        series = pd.Series(dict(sorted_contents)).apply(lambda x: ' '.join(x))
        return series
   

class MedicalChunkProcessor(OpenAIBatchAPI):

    instruction = MEDICAL_CHUNK_INSTRUCTION
    system_template = ""
    user_template = ""

    chunk_length = 1000
    model_name = "gpt-4o-mini"
    json_schema = None

    def __init__(self,
                 jsonl_path,
                 output_path,
                 description="Medical Chunk Processing"):

        super().__init__(jsonl_path, output_path, self.system_template,
                         self.user_template, self.json_schema, self.model_name, description)

    def get_system_and_user_templates(self, chunk) -> Tuple[str, str]:
        system = self.instruction
        user = chunk
        return system, user

    def create_batch(self, df_paths):
        batch_jsons = []
        for path in sorted(df_paths):
            df = pd.read_csv(path)
            df_id = path.split('/')[-1].strip('.csv')
            for row_idx, row in df.iterrows():
                document = row['TEXT']
                for chunk_idx, chunk_start in enumerate(range(0, len(document), self.chunk_length)):
                    json_batch = self.create_batch_json(
                        f"{df_id}_{row_idx}_{chunk_idx}", row['TEXT'][chunk_start: chunk_start+self.chunk_length])
                    batch_jsons.append(json_batch)

        self.save_batch_jsons(batch_jsons, self.jsonl_path)
        batch_object_id = self.start_batch_job(self.jsonl_path)
        return batch_object_id

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


class MedicalRefiner(OpenAIBatchAPI):
    instruction = MEDICAL_REFINER_INSTRUCTION
    system_template = ""
    user_template = MEDICAL_REFINER_TEMPLATE

    chunk_length = 1000
    model_name = "gpt-4o-mini"
    json_schema = None

    def __init__(self,
                 jsonl_path,
                 output_path,
                 description="Medical Chunk Processing"):

        super().__init__(jsonl_path, output_path, self.system_template,
                         self.user_template, self.json_schema, self.model_name, description)

    def get_system_and_user_templates(self, input, summary) -> Tuple[str, str]:
        system = self.instruction
        user = self.user_template.format(input=input, summary=summary)
        return system, user

    def parse_output(self, output_path: str) -> list[dict]:
        parsed_contents = defaultdict(list)  # {row_idx, chunk_idx}
        errors = []
        with open(output_path, 'r') as f:
            for line in f:
                output = json.loads(line)
                custom_id = output['custom_id']
                row_idx = custom_id
                chunk_idx = custom_id
                # row_idx, chunk_idx = custom_id.split('_')
                # row_idx = int(row_idx)
                # chunk_idx = int(chunk_idx)
                try:
                    content = literal_eval(
                        output['response']['body']['choices'][0]['message']['content'])
                    parsed_contents[row_idx].append((chunk_idx, content))
                except:
                    errors.append((row_idx, chunk_idx))

        formatted_contents = {}
        for row_idx, data in tqdm(parsed_contents.items()):
            json_data = []
            # ensure it is sorted.
            data = sorted(data, key=lambda x: x[0])
            for chunk_idx, json_summary in data:
                # for chunk_idx, text in json_summary.items():
                text = json_summary['overall_summary']
                if text != '':
                    json_data.append(text.capitalize())
            formatted_contents[row_idx] = json_data
        sorted_contents = sorted(formatted_contents.items())
        series = pd.Series(dict(sorted_contents)).apply(lambda x: ' '.join(x))
        return series
