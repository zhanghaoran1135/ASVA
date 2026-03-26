from contextlib import nullcontext
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, RobertaForMaskedLM

from asva.data.cache_utils import ensure_dir, stable_hash

LOGGER = logging.getLogger(__name__)


class CodeBERTEncoder(nn.Module):
    def __init__(
        self,
        model_path,
        cache_dir=None,
        use_embedding_cache=False,
        enable_gradient_checkpointing=False,
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.mlm_model = RobertaForMaskedLM.from_pretrained(self.model_path, local_files_only=True)
        self.backbone = self.mlm_model.roberta
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.use_embedding_cache = use_embedding_cache
        self.cache_dir = ensure_dir(Path(cache_dir)) if cache_dir else None
        self.last_tokenized = None
        self.gradient_checkpointing_enabled = False
        if enable_gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _cache_allowed(self):
        if not self.use_embedding_cache or self.cache_dir is None:
            return False
        return not self.training or not any(param.requires_grad for param in self.backbone.parameters())

    def _cache_path(self, text, max_length):
        return self.cache_dir / f"{stable_hash([text, max_length])}.pt"

    def _pooled_cache_path(self, text, max_length):
        return self.cache_dir / f"{stable_hash(['pooled', text, max_length])}.pt"

    @staticmethod
    def masked_mean_pool(hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def gradient_checkpointing_enable(self):
        if hasattr(self.mlm_model, "gradient_checkpointing_enable"):
            self.mlm_model.gradient_checkpointing_enable()
        elif hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        elif hasattr(self.backbone, "encoder"):
            setattr(self.backbone.encoder, "gradient_checkpointing", True)
        self.gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.mlm_model, "gradient_checkpointing_disable"):
            self.mlm_model.gradient_checkpointing_disable()
        elif hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()
        elif hasattr(self.backbone, "encoder"):
            setattr(self.backbone.encoder, "gradient_checkpointing", False)
        self.gradient_checkpointing_enabled = False

    def tokenize(self, texts, max_length, device):
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in encoded.items()}

    def encode_texts(self, texts, max_length, device):
        if not texts:
            empty_hidden = torch.zeros(0, 1, self.hidden_size, device=device)
            empty_mask = torch.zeros(0, 1, dtype=torch.long, device=device)
            empty_pool = torch.zeros(0, self.hidden_size, device=device)
            return {
                "hidden_states": empty_hidden,
                "attention_mask": empty_mask,
                "pooled": empty_pool,
                "input_ids": empty_mask,
            }
        if not self._cache_allowed():
            tokenized = self.tokenize(texts, max_length, device)
            outputs = self.backbone(**tokenized)
            pooled = self.masked_mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
            self.last_tokenized = tokenized
            return {
                "hidden_states": outputs.last_hidden_state,
                "attention_mask": tokenized["attention_mask"],
                "pooled": pooled,
                "input_ids": tokenized["input_ids"],
            }
        cached = {}
        missing_indices = []
        missing_texts = []
        for idx, text in enumerate(texts):
            cache_path = self._cache_path(text, max_length)
            if cache_path.exists():
                cached[idx] = torch.load(cache_path, map_location=device)
            else:
                missing_indices.append(idx)
                missing_texts.append(text)
        if missing_texts:
            with torch.no_grad():
                tokenized = self.tokenize(missing_texts, max_length, device)
                outputs = self.backbone(**tokenized)
                pooled = self.masked_mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
            for local_idx, global_idx in enumerate(missing_indices):
                item = {
                    "hidden_states": outputs.last_hidden_state[local_idx].detach().cpu(),
                    "attention_mask": tokenized["attention_mask"][local_idx].detach().cpu(),
                    "pooled": pooled[local_idx].detach().cpu(),
                    "input_ids": tokenized["input_ids"][local_idx].detach().cpu(),
                }
                torch.save(item, self._cache_path(texts[global_idx], max_length))
                cached[global_idx] = {key: value.to(device) for key, value in item.items()}
        ordered = [cached[idx] for idx in range(len(texts))]
        hidden_states = torch.stack([item["hidden_states"] for item in ordered], dim=0).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in ordered], dim=0).to(device)
        pooled = torch.stack([item["pooled"] for item in ordered], dim=0).to(device)
        input_ids = torch.stack([item["input_ids"] for item in ordered], dim=0).to(device)
        self.last_tokenized = {"attention_mask": attention_mask, "input_ids": input_ids}
        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "pooled": pooled,
            "input_ids": input_ids,
        }

    def encode_pooled_texts(
        self,
        texts,
        max_length,
        device,
        chunk_size=None,
        force_no_grad=False,
    ):
        if not texts:
            return torch.zeros(0, self.hidden_size, device=device)
        if chunk_size is None or chunk_size <= 0 or len(texts) <= chunk_size:
            return self._encode_pooled_batch(texts, max_length=max_length, device=device, force_no_grad=force_no_grad)
        pooled_chunks = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            pooled_chunks.append(
                self._encode_pooled_batch(chunk, max_length=max_length, device=device, force_no_grad=force_no_grad)
            )
        return torch.cat(pooled_chunks, dim=0)

    def _encode_pooled_batch(
        self,
        texts,
        max_length,
        device,
        force_no_grad=False,
    ):
        restore_checkpointing = force_no_grad and self.gradient_checkpointing_enabled
        if restore_checkpointing:
            self.gradient_checkpointing_disable()
        try:
            if self._cache_allowed() and force_no_grad:
                cached = {}
                missing_indices = []
                missing_texts = []
                for idx, text in enumerate(texts):
                    cache_path = self._pooled_cache_path(text, max_length)
                    if cache_path.exists():
                        cached[idx] = torch.load(cache_path, map_location=device)
                    else:
                        missing_indices.append(idx)
                        missing_texts.append(text)
                if missing_texts:
                    tokenized = self.tokenize(missing_texts, max_length, device)
                    with torch.no_grad():
                        outputs = self.backbone(**tokenized)
                        pooled = self.masked_mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
                    for local_idx, global_idx in enumerate(missing_indices):
                        item = pooled[local_idx].detach().cpu()
                        torch.save(item, self._pooled_cache_path(texts[global_idx], max_length))
                        cached[global_idx] = item.to(device)
                return torch.stack([cached[idx].to(device) for idx in range(len(texts))], dim=0)
            tokenized = self.tokenize(texts, max_length, device)
            grad_context = torch.no_grad if force_no_grad else nullcontext
            with grad_context():
                outputs = self.backbone(**tokenized)
                pooled = self.masked_mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
            return pooled
        finally:
            if restore_checkpointing:
                self.gradient_checkpointing_enable()

    def mask_inputs(
        self,
        input_ids,
        special_tokens_mask=None,
        mlm_probability=0.15,
    ):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability, device=input_ids.device)
        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
                    for val in labels
                ],
                dtype=torch.bool,
                device=input_ids.device,
            )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        valid_token_mask = ~special_tokens_mask
        for row_idx in range(masked_indices.size(0)):
            if masked_indices[row_idx].any():
                continue
            candidate_indices = torch.nonzero(valid_token_mask[row_idx], as_tuple=False).flatten()
            if candidate_indices.numel() == 0:
                continue
            masked_indices[row_idx, int(candidate_indices[0].item())] = True
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_indices
        input_ids = input_ids.clone()
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=input_ids.device)
        input_ids[indices_random] = random_words[indices_random]
        return input_ids, labels

    def compute_mlm_loss(self, input_ids, attention_mask, mlm_probability=0.15):
        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(ids.tolist(), already_has_special_tokens=True)
                for ids in input_ids
            ],
            dtype=torch.bool,
            device=input_ids.device,
        )
        special_tokens_mask |= attention_mask.eq(0)
        masked_inputs, labels = self.mask_inputs(input_ids, special_tokens_mask=special_tokens_mask, mlm_probability=mlm_probability)
        if not (labels != -100).any():
            return torch.zeros((), dtype=torch.float32, device=input_ids.device)
        outputs = self.mlm_model(
            input_ids=masked_inputs,
            attention_mask=attention_mask,
            labels=labels,
        )
        return torch.nan_to_num(outputs.loss, nan=0.0, posinf=0.0, neginf=0.0)
