import torch
import torch.nn as nn
from utils.tokens import completions_only_labels, convert_prompt_to_tokens


class HybridModel(nn.Module):

    def __init__(self, tokenizer, llm_model, max_length):
        super(HybridModel, self).__init__()

        self.ignore_index = -100
        self.response_token_ids = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.max_length = max_length

    def convert_tokens_to_embedding(self, llm_model, tokens):
        return llm_model.model.embed_tokens(tokens.input_ids)

    def get_temporal_token_indexes(self, tokens, input_dates, max_length) -> list[list[int]]:
        """
        Find indexes of where to insert temporal tokens (right before each input dates)
        from input_ids of texts
        """
        input_ids = tokens.input_ids
        input_date_ids = [[id[1:] for id in self.tokenizer(
            date).input_ids] for date in input_dates]

        all_temporal_indexes = []
        for b in range(len(input_dates)):
            curr_idx = 0
            temporal_indexes = []

            # assume all date lengths are equal
            date_length = len(input_date_ids[b][curr_idx])

            # start from non-padded text for faster search
            for i in range(0, max_length - date_length - 1):
                if torch.equal(input_ids[b, i:i+date_length], torch.tensor(input_date_ids[b][curr_idx], device=input_ids.device)):
                    temporal_indexes.append(i)
                    curr_idx += 1
                    if curr_idx == len(input_dates[0]):
                        break
            all_temporal_indexes.append(temporal_indexes)
        return all_temporal_indexes

    # insert temporal token from hidden_state
    def insert_temporal_token(self, embeddings, hidden_state, temporal_token_indexes) -> torch.Tensor:
        assert hidden_state.shape[-1] == embeddings.shape[-1], 'hidden state dimension is not equal to text embedding shape'
        batch_size = len(temporal_token_indexes)

        patched_embeddings = torch.zeros(
            batch_size, embeddings.shape[1] + hidden_state.shape[1], embeddings.shape[-1], device=embeddings.device)
        for b in range(batch_size):
            curr_idx = 0
            curr_pos = 0
            for idx in temporal_token_indexes[b]:
                # first embedding
                patched_embeddings[b,
                                   curr_pos:idx] = embeddings[b, curr_pos:idx]
                # insert temporal token
                patched_embeddings[b, idx:idx+1] = hidden_state[b][curr_idx]
                curr_pos = idx + 1  # update pos to immediately after temporal token
                curr_idx += 1
            patched_embeddings[b, curr_pos:] = embeddings[b,
                                                          curr_pos - curr_idx:]  # rest of embedding
        return patched_embeddings

    # patch attention mask
    def patch_attention_mask(self, attention_mask, input_window, padding_side='right') -> torch.Tensor:
        # attention mask has shape [B, D]. Insert torch.ones
        ones = torch.ones(attention_mask.size(
            0), input_window, device=attention_mask.device)
        if padding_side == 'right':  # for training
            patched_attention_mask = torch.cat((ones, attention_mask), dim=1)
        elif padding_side == 'left':  # for inference
            patched_attention_mask = torch.cat((attention_mask, ones), dim=1)
        return patched_attention_mask
    # patch prediction labels

    def patch_labels(self, labels, input_window, ignore_index=-100) -> torch.Tensor:
        """
        Insert input_window number of ignore indexes in the beginning.

        Assume temporal tokens are all added before the assistant prompt
        """
        batch_size, seq_len = labels.shape
        ignore_indexes = torch.full(
            (batch_size, input_window), ignore_index, device=labels.device)
        patched_labels = torch.cat((ignore_indexes, labels), dim=1)
        return patched_labels

    def forward(self, prompts, input_dates, hidden_state, inference=False):
        """
        In prompts, at input_dates positions, insert hidden_state
        and patch attention mask and prediction only labels
        and return output and hybrid_labels
        """
        tokens = convert_prompt_to_tokens(
            self.tokenizer, prompts, self.max_length, self.llm_model.device)
        embeddings = self.convert_tokens_to_embedding(
            self.llm_model.model, tokens)

        temporal_token_indexes = self.get_temporal_token_indexes(
            tokens, input_dates, self.max_length)
        hybrid_embeddings = self.insert_temporal_token(
            embeddings, hidden_state, temporal_token_indexes)
        hybrid_attention_mask = self.patch_attention_mask(
            tokens.attention_mask, len(input_dates[0]), padding_side='left' if inference else 'right')

        if not inference:
            labels = completions_only_labels(
                tokens, self.response_token_ids, self.ignore_index)
            hybrid_labels = self.patch_labels(
                labels, len(input_dates[0]), self.ignore_index)
            output = self.llm_model(
                inputs_embeds=hybrid_embeddings, attention_mask=hybrid_attention_mask)
            return output, hybrid_labels
        else:
            output_ids = self.llm_model.generate(inputs_embeds=hybrid_embeddings.to(self.llm_model.dtype),
                                                 attention_mask=hybrid_attention_mask.to(
                                                     self.llm_model.dtype),
                                                 max_new_tokens=self.max_length)
            output_texts = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenizationspaces=True)
            return output_texts
