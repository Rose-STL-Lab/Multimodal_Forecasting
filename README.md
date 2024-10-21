# Multimodal Forecasting



## Prerequisites

Before you begin, ensure you have accounts set up on:
- [Weights & Biases (wandb)](https://wandb.ai/)
- [Hugging Face](https://huggingface.co/)

## Project Setup

1. **Clone the repository:**
   ```
   git clone git@github.com:Rose-STL-Lab/Multimodal_Forecasting.git
   cd Multimodal_Forecasting
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Configure wandb and Hugging Face credentials:**
   Follow the documentation for each platform to set up your API keys and authentication.

## Workflow

### 1. Data Preparation

Upload your dataset to Hugging Face. Example:
https://huggingface.co/datasets/Howard881010/climate-1day

### 2. Model Fine-tuning

We use [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning our models. Configuration files are located in `config/baseline/finetune/`.

### 3. Prepare Dataset for Inference and Evaluation

Since we will have four different cases (text2text, textTime2textTime, textTime2text, textTime2time), so we create a new huggingface dataset to do this. For example,  
   - Input: (input_text + instruction 1)  
   - Output: pred_output_case1  
   - Ground Truth: output_text  
   
Those columns will be used to generate, save, and evaluate the result for text2text case.  
Example dataset: https://huggingface.co/datasets/Howard881010/climate-1day-finetuned

### 4. Inference and Evaluation

Run inference using either the fine-tuned model or a pre-trained model. Results will be saved on Hugging Face, and evaluations will be logged to wandb.

## Usage

To run the main script:

For Hybrid model
```
# run mlp pretraining stage
python -m src.hybrid -i 1 -o 1 -ms --config='config/medical_bge.yaml' &&

# evaluate mlp pretraining stage
python -m src.hybrid -i 1 -o 1 -ms -t --config='config/medical_bge.yaml' &&

# run end-to-end finetuning stage
python -m src.hybrid -i 1 -o 1 -hs --config='config/medical_bge.yaml' &&

# make inference on end-to-end finetuning stage
python -m src.hybrid -i 1 -o 1 -hs -t --config='config/medical_bge.yaml' &&

# evaluate end-to-end finetuning stage
python -m src.evaluate -i 1 -o 1 -hs --config='config/medical_bge.yaml'
```

For fine-tuned model:
```
python baseline_model/multimodal.py -i 1 -o 1 --config="config/baseline/climate.yml"
```
For pre-trained model and zeroshot case:
```
python baseline_model/multimodal_zeroshot.py -i 1 -o 1 --config="config/baseline/climate.yml"
```
For pre-trained model and in-context case:
```
python baseline_model/multimodal_inContext.py -i 1 -o 1 --config="config/baseline/climate.yml"
```
For nlinear model:
```
python baseline_model/nlinear.py -i 1 -o 1 --config="config/baseline/climate.yml"
```
For nlinear model with text embedding:
```
python baseline_model/nlinear_textEmbedding.py -i 1 -o 1 --config="config/baseline/climate.yml"
```


Parameters:
- `-i`: Input parameter
- `-o`: Output parameter
- `--config`: Path to the configuration file

## Configuration

Modify the `config/baseline/climate.yml` file to adjust model parameters, data paths, and other settings.

## Results

- Model outputs and datasets will be saved to your Hugging Face account.
- Evaluation metrics and experiment tracking will be available on your wandb dashboard.

## Contributing

[Include guidelines for contributing to the project, if applicable]

## License

[Specify the license under which this project is released]

## Contact

[Your contact information or how to reach the project maintainers]