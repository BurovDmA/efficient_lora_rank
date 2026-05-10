import inspect
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase


def _chat_template_kwargs(tokenizer: PreTrainedTokenizerBase) -> dict:
    """Дополнительные параметры chat template для конкретного токенизатора."""
    if not hasattr(tokenizer, "_extra_chat_kwargs"):
        extra = {}
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)
            if "enable_thinking" in sig.parameters:
                extra["enable_thinking"] = False
        except (ValueError, TypeError):
            try:
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "test"}],
                    tokenize=False,
                    enable_thinking=False,
                )
                extra["enable_thinking"] = False
            except (TypeError, Exception):
                pass
        tokenizer._extra_chat_kwargs = extra
    return tokenizer._extra_chat_kwargs


class SFTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._extra_kwargs = _chat_template_kwargs(tokenizer)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        question = str(self.df.iloc[idx]["question"])
        answer = str(self.df.iloc[idx]["answer"])

        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            **self._extra_kwargs,
        )

        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
            **self._extra_kwargs,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        full_enc = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        labels = input_ids.copy()
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[dict], tokenizer: PreTrainedTokenizerBase) -> dict:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]

    pad = lambda seqs, val: torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=val)

    return {
        "input_ids": pad(input_ids, tokenizer.pad_token_id),
        "attention_mask": pad(attention_mask, 0),
        "labels": pad(labels, -100),
    }


def _full_text(question: str, answer: str, tokenizer: PreTrainedTokenizerBase) -> str:
    extra = _chat_template_kwargs(tokenizer)
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
        **extra,
    )


def filter_df_by_max_len(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    q_col: str = "question",
    a_col: str = "answer",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lengths, keep_mask = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Token length scan"):
        text = _full_text(str(row[q_col]), str(row[a_col]), tokenizer)
        length = len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
        lengths.append(length)
        keep_mask.append(length <= max_length)

    out = df.copy()
    out["tok_len"] = lengths

    filtered = out[keep_mask].reset_index(drop=True)
    dropped = out[[not x for x in keep_mask]].reset_index(drop=True)

    kept_pct = len(filtered) / max(len(out), 1)
    print(
        f"Kept: {len(filtered)}/{len(out)} ({kept_pct:.2%}) | "
        f"Dropped: {len(dropped)} | "
        f"max tok_len: {out['tok_len'].max()}"
    )
    return filtered, dropped
