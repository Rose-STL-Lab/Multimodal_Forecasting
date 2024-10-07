from sentence_transformers import SentenceTransformer
from glob import glob
from huggingface_hub import login
import pandas as pd
import os
from utils.dataset import get_embedding_loader, get_embedding_loader_from_list, apply_chat_template
from utils.config import load_config
from utils.tokens import convert_prompt_to_tokens, completions_only_labels, get_masked_llm_loss
import torch
from tqdm import tqdm, trange
from loguru import logger
from transformers import (
    BitsAndBytesConfig,
    set_seed,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
import torch.nn.functional as F
from model.hybrid import HybridModel
from model.multihead_mlp import MultiHeadMLP
import wandb
import argparse
from dotenv import load_dotenv
load_dotenv('.env')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiHead MLP Configuration")
    parser.add_argument('-i', '--input_window', type=int,
                        help='Input window size')
    parser.add_argument('-o', '--output_window', type=int,
                        help='Output window size')
    parser.add_argument('-t', '--inference',
                        action='store_true', help='Inference')
    parser.add_argument('-ms', '--mlp_stage', action='store_true',
                        help='train mlp only')
    parser.add_argument('-hs', '--hybrid_stage', action='store_true',
                        help='train mlp and llm at the same time')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='debug by breaking loops')
    parser.add_argument('-c', '--config', type=str,
                        help='debug by breaking loops')
    parser.add_argument('-r', '--results_dir', type=str,
                        help='debug by breaking loops')

    args = parser.parse_args()
    if args.config:
        config_path = args.config
    else:
        config_path = 'config/config.yaml'
    cfg = load_config(config_path)
    set_seed(cfg['seed'])

    llm_cfg = cfg['llm']
    mlp_cfg = cfg['mlp']
    if args.input_window:
        cfg['input_window'] = args.input_window
    if args.output_window:
        cfg['output_window'] = args.output_window
    if args.debug:
        cfg['debug'] = True
    else:
        cfg['debug'] = False

    if args.results_dir:
        cfg['results_dir'] = args.results_dir

    if args.mlp_stage:
        current_stage = cfg['mlp_dir']
    elif args.hybrid_stage:
        current_stage = cfg['hybrid_dir']
    assert current_stage is not None, "Specify a stage to run."

    # ------------ add loggers ------------
    exp_dir = f"{cfg['results_dir']}/{cfg['input_window']}_{cfg['output_window']}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, current_stage), exist_ok=True)
    log_name = "eval.log" if args.inference else "train.log"
    logger.add(f"{exp_dir}/{current_stage}/{log_name}", rotation="1000 MB")
    if cfg['debug'] or args.inference:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(name=f"{current_stage}-{cfg['input_window']}-{cfg['output_window']}",
               project='multimodal_forecasting')
    logger.info(cfg)

    input_window = cfg['input_window']
    output_window = cfg['output_window']

    instruction_template = cfg['instruction_template'].format(
        input_window=input_window, output_window=output_window)
    date_template = cfg['date_template']
    text_template = cfg['text_template']

    # login(token=os.environ["HUGGINGFACE_API"])
    # need to use transformers model for .generate
    if current_stage == cfg['hybrid_dir'] and args.inference:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_cfg['model_name'], padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

        if llm_cfg['load_in_4bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=llm_cfg['load_in_8bit'])

        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_cfg['model_name'],
            quantization_config=quantization_config,
            # attn_implementation="flash_attention_2",
            device_map="cuda")

        if current_stage == cfg['hybrid_dir']:
            model_path = os.path.join(exp_dir, cfg['hybrid_dir'], 'best')
        llm_model = PeftModel.from_pretrained(
            llm_model, model_path, device_map="cuda")
    else:
        assert not (llm_cfg['load_in_8bit'] and llm_cfg['load_in_4bit']
                    ), "Load in either 4bit or 8bit or full precision."
        unsloth_import_retries = 5
        for attempt in range(unsloth_import_retries):
            try:
                from unsloth import FastLanguageModel
                model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
                llm_model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    device_map="cuda",
                    load_in_8bit=llm_cfg['load_in_8bit'],
                    load_in_4bit=llm_cfg['load_in_4bit']
                )
                # keep model.model call consistent for mlp_dir
                if current_stage == cfg['mlp_dir'] or (not args.inference and current_stage == cfg['hybrid_dir']):
                    llm_model = FastLanguageModel.get_peft_model(
                        llm_model,
                        r=llm_cfg['lora_r'],
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj",],
                        lora_alpha=llm_cfg['lora_alpha'],
                        lora_dropout=llm_cfg['lora_dropout'],
                        bias="none",
                        use_gradient_checkpointing="unsloth",
                        random_state=cfg['seed'],
                        use_rslora=False,
                        loftq_config=None,
                    )
                elif current_stage == cfg['hybrid_dir']:  # inference
                    lora_path = os.path.join(
                        exp_dir, cfg['hybrid_dir'], 'best')
                    llm_model = PeftModel.from_pretrained(llm_model, lora_path)
                if current_stage == cfg['mlp_dir'] or args.inference:
                    FastLanguageModel.for_inference(llm_model)
                else:
                    FastLanguageModel.for_training(llm_model, tokenizer)
                break
            except RuntimeError as e:
                if attempt == unsloth_import_retries - 1:
                    logger.error(f"Unsloth import failed")
                    raise

    device = torch.device('cuda')
    if current_stage == cfg['mlp_dir']:
        batch_size = mlp_cfg['batch_size']
    else:
        batch_size = llm_cfg['batch_size']

    if mlp_cfg.get('embd_model', None) is not None:
        embd_model = SentenceTransformer(mlp_cfg['embd_model'])
    else:
        embd_model = None

    if not os.path.isdir(cfg['dataset_path']):
        df = pd.read_csv(cfg['dataset_path'])
        train_loader, valid_loader, test_loader, \
            minmax_fn, de_minmax_fn = get_embedding_loader(
                df, input_window, output_window, tokenizer, llm_model.model, batch_size, device, embd_model=embd_model)
    else:
        df_list = sorted(glob(cfg['dataset_path'] + "/*"))
        train_loader, valid_loader, test_loader, \
            minmax_fn, de_minmax_fn = get_embedding_loader_from_list(
                df_list, input_window, output_window, tokenizer, llm_model.model, batch_size, device, embd_model=embd_model)

    # get max token
    max_length = 0
    for data in tqdm(train_loader, total=len(train_loader), desc="calculating max length..."):
        input_times, output_times, input_texts, output_texts, input_dates, output_dates, input_embeddings = data
        prompts = apply_chat_template(instruction_template, date_template, text_template, tokenizer,
                                      input_texts, input_dates, output_dates, output_texts)
        with torch.no_grad():
            batch_max_length = max(
                [len(tokenizer(prompt, padding=False).input_ids) for prompt in prompts])
        max_length = max(max_length, batch_max_length)

    hybrid_model = HybridModel(tokenizer, llm_model, max_length=max_length)

    if current_stage != cfg['llm_dir']:
        mlp_model = MultiHeadMLP(cfg, device).to(device)
        if (current_stage == cfg['mlp_dir'] and args.inference) or \
                (current_stage == cfg['hybrid_dir'] and not args.inference):
            mlp_model.load_state_dict(torch.load(
                os.path.join(exp_dir, cfg['mlp_dir'], 'best')))
        elif (current_stage == cfg['hybrid_dir'] and args.inference):
            mlp_model.load_state_dict(torch.load(
                os.path.join(exp_dir, cfg['hybrid_dir'], 'mlp_best')))

    if current_stage == cfg['mlp_dir']:
        train_cfg = mlp_cfg
    else:
        train_cfg = llm_cfg
    if current_stage == cfg['mlp_dir']:
        parameters = mlp_model.parameters()
    elif current_stage == cfg['hybrid_dir']:
        parameters = list(llm_model.parameters()) + \
            list(mlp_model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=train_cfg['learning_rate'])

    if train_cfg['lr_scheduler'] == "linear":
        get_lr_scheduler = get_linear_schedule_with_warmup
    elif train_cfg['lr_scheduler'] == "cosine":
        get_lr_scheduler = get_cosine_schedule_with_warmup

    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        num_warmup_steps=train_cfg['num_warm_up_steps'],
        num_training_steps=(len(train_loader) * train_cfg['epochs'])
    )

    best_loss = float('inf')
    best_val_epoch = 0
    if not args.inference:
        # --------- train valid split ------------
        epochs = train_cfg['epochs']
        if current_stage == cfg['mlp_dir']:
            patience = train_cfg['patience']
            patience_counter = 0
        for epoch in (pbar := trange(epochs, desc="Epochs")):
            train_mse_loss = 0
            train_nll_loss = 0
            valid_mse_loss = 0
            valid_nll_loss = 0

            mlp_model.train()
            if current_stage == cfg['hybrid_dir']:
                llm_model.train()

            for b, data in tqdm(enumerate(train_loader), desc="Training Batches", total=len(train_loader), leave=False):
                # format input and output time+text
                input_times, output_times, input_texts, output_texts, input_dates, output_dates, input_embeddings = data
                prompts = apply_chat_template(instruction_template, date_template, text_template, tokenizer,
                                              input_texts, input_dates, output_dates, output_texts)
                input_times = minmax_fn(input_times)
                output_times = minmax_fn(output_times)

                # forward time
                time_output, hidden_state = mlp_model(
                    input_times, input_embeddings, return_hidden_state=True)
                mse_loss = F.mse_loss(time_output, output_times)

                # forward text
                if current_stage == cfg['mlp_dir']:
                    nll_loss = torch.tensor(
                        0.0, device=device, requires_grad=True)
                else:
                    text_outputs, labels = hybrid_model(
                        prompts, input_dates, hidden_state)
                    nll_loss = get_masked_llm_loss(text_outputs.logits, labels)

                if current_stage == cfg['mlp_dir']:
                    loss = mse_loss * cfg['llm']['mlp_weight']
                else:
                    loss = nll_loss + mse_loss * cfg['llm']['mlp_weight']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                train_mse_loss += mse_loss.item()
                train_nll_loss += nll_loss.item()

                train_log = {
                    "epoch": epoch,
                    "batch": b,
                    "train_nll_loss": nll_loss.item(),
                    "train_mse_loss": mse_loss.item(),
                    "train_loss": loss.item()
                }

                if current_stage == cfg['hybrid_dir']:
                    wandb.log(train_log)
                    logger.info(train_log)
                    if b % int(len(train_loader) / 10) == 0:
                        predictions = torch.argmax(
                            text_outputs.logits[0], dim=-1)
                        training_prediction = tokenizer.decode(
                            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        logger.info(
                            f"Training prediction: {training_prediction[:200]}")
                pbar.set_description(
                    f"{current_stage} Epoch {epoch} | Batch {b} / {len(train_loader)} mse={mse_loss.item():.4f} nll={nll_loss.item():.4f}")

                # TODO: save checkpoint
                if cfg['debug']:
                    break

            train_mse_loss /= len(train_loader)
            train_nll_loss /= len(train_loader)
            avg_train_log = {"train_mse_loss": train_mse_loss,
                             "train_nll_loss": train_nll_loss, "epoch": epoch}
            if current_stage == cfg['hybrid_dir']:
                wandb.log(avg_train_log)
                logger.info(
                    f"{current_stage} Epoch {epoch} Train MSE = {train_mse_loss:.4f} NLL = {train_nll_loss}")

            mlp_model.eval()
            if current_stage == cfg['hybrid_dir']:
                llm_model.eval()

            with torch.no_grad():
                for b, data in tqdm(enumerate(valid_loader), desc="Validing Batches", total=len(valid_loader), leave=False):
                    input_times, output_times, input_texts, output_texts, input_dates, output_dates, input_embeddings = data
                    prompts = apply_chat_template(instruction_template, date_template, text_template, tokenizer,
                                                  input_texts, input_dates, output_dates, output_texts)
                    input_times = minmax_fn(input_times)
                    output_times = minmax_fn(output_times)
                    time_output, hidden_state = mlp_model(
                        input_times, input_embeddings, return_hidden_state=True)
                    mse_loss = F.mse_loss(time_output, output_times)

                    if current_stage == cfg['mlp_dir']:
                        nll_loss = torch.tensor(0.0, device=device)
                    else:
                        text_outputs, labels = hybrid_model(
                            prompts, input_dates, hidden_state)
                        nll_loss = get_masked_llm_loss(
                            text_outputs.logits, labels)

                    loss = mse_loss + nll_loss
                    valid_mse_loss += mse_loss.item()
                    valid_nll_loss += nll_loss.item()

                    valid_log = {
                        "epoch": epoch,
                        "batch": b,
                        "valid_nll_loss": nll_loss.item(),
                        "valid_mse_loss": mse_loss.item(),
                        "valid_loss": loss.item(),
                    }
                    if current_stage == cfg['hybrid_dir']:
                        wandb.log(valid_log)
                        logger.info(valid_log)
                        if b % int(len(valid_loader) / 10) == 0:
                            predictions = torch.argmax(
                                text_outputs.logits[0], dim=-1)
                            validation_predictions = tokenizer.decode(
                                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            logger.info(
                                f"Validation prediction: {validation_predictions[:200]}")
                    pbar.set_description(
                        f"{current_stage} Epoch {epoch} | Batch {b} / {len(valid_loader)} mse={mse_loss.item():.4f} nll={nll_loss.item():.4f}")

                    if cfg['debug']:
                        break

            valid_mse_loss /= len(valid_loader)
            valid_nll_loss /= len(valid_loader)
            avg_valid_log = {"valid_mse_loss": valid_mse_loss,
                             "valid_nll_loss": valid_nll_loss, "epoch": epoch}
            if current_stage == cfg['hybrid_dir']:
                wandb.log(avg_valid_log)
                logger.info(
                    f"{current_stage} Epoch {epoch} Valid MSE = {valid_mse_loss:.4f} NLL = {valid_nll_loss}")

            # Save the best model
            if valid_mse_loss + valid_nll_loss < best_loss:
                logger.info(
                    f"Saving {current_stage} best model at epoch {epoch} mse={valid_mse_loss} nll={valid_nll_loss}")
                best_loss = valid_mse_loss + valid_nll_loss
                if current_stage == cfg['mlp_dir']:
                    torch.save(mlp_model.state_dict(),
                               os.path.join(exp_dir, cfg['mlp_dir'], 'best'))
                    patience_counter = 0
                else:
                    llm_model.save_pretrained(
                        os.path.join(exp_dir, current_stage, 'best'))
                    torch.save(mlp_model.state_dict(), os.path.join(
                        exp_dir, current_stage, 'mlp_best'))
            else:
                if current_stage == cfg['mlp_dir']:
                    patience_counter += 1
            if current_stage == cfg['mlp_dir'] and patience_counter >= patience:
                break

            if cfg['debug']:
                break
    else:
        # --------- inference ------------
        mlp_model.eval()
        if current_stage == cfg['hybrid_dir']:
            llm_model.eval()
        results = []

        with torch.no_grad():
            if current_stage == cfg['mlp_dir']:
                all_pred_times = []
                all_output_times = []
                for b, data in tqdm(enumerate(test_loader), desc="Predicting Batches", leave=False, total=len(test_loader)):
                    input_times, output_times, input_texts, output_texts, input_dates, output_dates, input_embeddings = data
                    input_times = minmax_fn(input_times)
                    pred_times = mlp_model(input_times, input_embeddings)
                    all_pred_times.append(de_minmax_fn(pred_times))
                    all_output_times.append(output_times)
                all_pred_times = torch.vstack(all_pred_times)
                all_output_times = torch.vstack(all_output_times)
                mse = torch.mean(torch.square(
                    all_pred_times - all_output_times))
                rmse = torch.sqrt(mse)
                logger.info(f"{current_stage} test RMSE={rmse.item():.4f}")
            else:
                for b, data in tqdm(enumerate(test_loader), desc="Predicting Batches", leave=False, total=len(test_loader)):
                    input_times, output_times, input_texts, output_texts, input_dates, output_dates, input_embeddings = data
                    prompts = apply_chat_template(instruction_template, date_template, text_template, tokenizer,
                                                  input_texts, input_dates, output_dates)
                    input_times = minmax_fn(input_times)
                    time_output, hidden_state = mlp_model(
                        input_times, input_embeddings, return_hidden_state=True)
                    text_outputs = hybrid_model(
                        prompts, input_dates, hidden_state, inference=True)
                    logger.info(f"batch {b}: {text_outputs[0][:200]}")

                    for i in range(len(input_times)):
                        results.append({
                            "input_times": de_minmax_fn(input_times[i]).cpu().numpy(),
                            "output_times": output_times[i].cpu().numpy(),
                            "input_texts": input_texts[i],
                            "output_texts": output_texts[i],
                            "input_dates": input_dates[i],
                            "output_dates": output_dates[i],
                            "pred_times": de_minmax_fn(time_output[i]).cpu().numpy(),
                            "pred_texts": text_outputs[i]
                        })
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(os.path.join(exp_dir, current_stage, 'results.csv'), index=False)

                    if cfg['debug']:
                        break
