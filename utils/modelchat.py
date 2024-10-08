from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import numpy as np
import torch
import time
from peft import PeftModel
import os

class ChatModel:
    def __init__(self, model_name, token, dataset, finetuned, case, device, window_size):
        self.model_name = model_name
        self.finetuned = finetuned
        self.token = token
        self.tokenizer = self.load_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = dataset
        self.case = case
        self.device = device
        self.window_size = window_size

    def load_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def load_tokenizer(self):
        raise NotImplementedError("Subclasses must implement this method")

    def chat(self, prompt):
        raise NotImplementedError("Subclasses must implement this method")

class LLMChatModel(ChatModel):
    def __init__(self, model_name, token, dataset, finetuned, case, device, window_size):
        super().__init__(model_name, token, dataset, finetuned, case, device, window_size)
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        # self.device = next(self.model.parameters()).device

    def load_model(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=self.token).to(self.device)

        if self.finetuned == True:
            return PeftModel.from_pretrained(base_model, f"Howard881010/{self.dataset}-{self.window_size}day")
        else: 
            return base_model
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
    def chat(self, prompt):
        new_prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False)
        model_inputs = self.tokenizer(
            new_prompt, return_tensors="pt", padding="longest")
        model_inputs = model_inputs.to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        # Generate text using the model
        with torch.no_grad():
            generate_ids = self.model.generate(
                model_inputs.input_ids, max_new_tokens=4096, eos_token_id=terminators, attention_mask=model_inputs.attention_mask)

        output = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True)

        return output
    
