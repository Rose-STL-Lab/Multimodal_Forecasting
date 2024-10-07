"""
Uploads dataset to $data.
Processes single csv (climate dataset) differently from multiple csvs (medical dataset).

Usage:
- python preprocess --dataset=$dataset --summarize_create # create job
- python preprocess --dataset=$dataset --check_status # check job
- python preprocess --dataset=$dataset --summarize_download # download job

- python preprocess --dataset=$dataset --summarize_create # create job
- python preprocess --dataset=$dataset --check_status # check job
- python preprocess --dataset=$dataset --summarize_download # download job
"""

import argparse
from batch_processor import ClimateChunkProcessor, ClimateRefiner, MedicalChunkProcessor, MedicalRefiner

def check_status():
    pass

def create_chunk_summarization_job():
    pass

def download_chunk_summarization_job():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset for summarization.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["climate", "medical"], help="Type of dataset.")
    parser.add_argument("--check_status", action="store_true", help="Check the status of a job.")
    parser.add_argument("--summarize_create", action="store_true", help="Create a summarization job.")
    parser.add_argument("--summarize_download", action="store_true", help="Download the results of a summarization job.")
    parser.add_argument("--refiner_create", action="store_true", help="Create a refiner job.")
    parser.add_argument("--refiner_download", action="store_true", help="Download the results of a refiner job.")
    args = parser.parse_args()

    jsonl_path = ""
    output_path = ""

    # Initialize chunk processor or refiner
    if args.dataset_type == "climate":
        if args.summarize_create or args.summarize_download:
            processor = ClimateChunkProcessor(jsonl_path, output_path)
        else:
            processor = ClimateRefiner(jsonl_path, output_path)
    elif args.dataset_type == "medical":
        if args.summarize_create or args.summarize_download:
            processor = MedicalChunkProcessor(jsonl_path, output_path)
        else:
            processor = MedicalRefiner(jsonl_path, output_path)
    

    if args.summarize_create:
        create_chunk_summarization_job(processor)
    elif args.check_status:
        check_status(processor)
    elif args.summarize_download:
        download_chunk_summarization_job(processor)