import torch
import torch.nn as nn
from tqdm import tqdm


def tokenize_split_days(texts: list[list], tokenizer, model, device, embd_model=None):
    """
    Splits each day into token embeddings
    params:
        texts: [N, T, ~L]

    returns:
        token_embeddings: [N, T, D]
    """
    embeddings = []
    with torch.no_grad():
        for text_window in tqdm(texts, total=len(texts), desc="Tokenizing texts..."):
            if embd_model is not None:
                embedding = torch.tensor([embd_model.encode(text) for text in text_window], device=device)
                embeddings.append(embedding)
            else:
                token_ids = [tokenizer(text, return_tensors='pt', padding=False).to(
                    device) for text in text_window]
                embedding = [model.model.embed_tokens(
                    token.input_ids).mean(1) for token in token_ids]
                embeddings.append(torch.stack(embedding).squeeze())
    return torch.stack(embeddings)


def convert_prompt_to_tokens(tokenizer, prompts, max_length, device):
    tokens = tokenizer(prompts,
                       max_length=max_length,
                       padding="max_length",
                       truncation=True,
                       return_tensors="pt").to(device)
    return tokens


def completions_only_labels(tokens, response_token_ids, ignore_index=-100) -> torch.Tensor:
    """
    For padding = right during training,
        tokens = [instruction, input, assistant_response, padding]
    returns 
        labels = [ignore_index, ignore_index, assistant_response, ignore_index]
    """
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask
    labels = input_ids.clone()
    response_token_ids_pt = torch.tensor(
        response_token_ids, device=input_ids.device)
    pad_start_idxs = attention_mask.sum(-1)

    batch_size, seq_len = input_ids.shape
    target_len = len(response_token_ids)

    for b in range(batch_size):
        for i in range(seq_len - target_len + 1):
            if torch.equal(input_ids[b, i:i + target_len], response_token_ids_pt):
                labels[b, :i] = ignore_index
                labels[b, pad_start_idxs[b]:] = ignore_index
                break

    return labels


def get_masked_llm_loss(logits, labels, epsilon=0.1, ignore_index=-100):
    """
    Code inspired from transformers import trainer_pt_utils # line 551, LabelSmoother

    Params:
        - logits of shape [B, max length, embedding]
        - labels of shape [B, max_length]

    Returns:
        - smoothed mask loss: scalar

    """
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous().unsqueeze(-1)

    log_probs = -nn.functional.log_softmax(logits, dim=-1)

    padding_mask = labels.eq(ignore_index)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    labels = torch.clamp(labels, min=0)
    nll_loss = log_probs.gather(dim=-1, index=labels)
    # works for fp16 input tensor too, by internally upcasting it to fp32
    smoothed_loss = log_probs.sum(
        dim=-1, keepdim=True, dtype=torch.float32)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements *
                                           log_probs.shape[-1])
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss
