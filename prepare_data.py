"""
Tokenize any dataset.

Tokenization style: flash-linear-attention style
- BOS token added at start of each document
- No EOS tokens between documents
- Pattern: [BOS] doc1 [BOS] doc2 [BOS] doc3 ...

Memory-efficient batched tokenization: flash-linear-attention style
- Processes docs in batches (batch_size=1000), not one by one
- Uses itertools.chain to flatten token lists lazily
- Creates fewer intermediate objects (1000 lists vs 1M lists)
- Only creates one final tensor instead of torch.cat() on millions of small tensors

Usage:
    python prepare_data.py

"""

import os
import numpy as np
import torch
from itertools import chain

from transformers import AutoTokenizer
from datasets import load_dataset

# use llama2 tokenizer
# tokenizer_name = "meta-llama/Llama-2-7b-hf"

tokenizer_name = "bicycleman15/tinystories-gpt4-clean-tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

save_path = "data/tinystories-gpt4-clean"


def preprocess_and_tokenize_batched(examples):
    """
    Batched tokenization (flash-linear-attention style).
    - Processes multiple documents at once
    - Chains all tokens together efficiently
    - Pattern: [BOS] doc1 [BOS] doc2 [BOS] doc3 ...
    """
    texts = examples["text"]
    # tokenize all texts in batch (returns list of token lists)
    input_ids = tokenizer(texts, return_attention_mask=False)["input_ids"]
    concatenated = list(chain(*input_ids))
    return {"input_ids": [concatenated]}


def tokenize_and_save(split, data, save_path, batch_size=1000):
    assert split in ["train", "test"]

    # remove all columns except input_ids after tokenization
    remove_columns = list(data.features.keys())

    tokenized_dataset = data.map(
        preprocess_and_tokenize_batched,
        batched=True,
        batch_size=batch_size,
        remove_columns=remove_columns,
        num_proc=32,
        desc=f"Tokenizing {split}",
    )

    # each row is a concatenated batch; chain them all into a numpy array
    # (avoids ~180GB Python list overhead — numpy stores raw int64 values at 8 bytes each)
    all_tokens = np.fromiter(chain(*tokenized_dataset["input_ids"]), dtype=np.int32)
    tokenized_dataset_tensor = torch.from_numpy(all_tokens)

    print(f"{split} tensor shape: {tokenized_dataset_tensor.shape}, dtype: {tokenized_dataset_tensor.dtype}")

    os.makedirs(save_path, exist_ok=True)

    final_save_path = os.path.join(save_path, f"{split}.pt")

    print(f"Saving {split} tensor to:", final_save_path)
    torch.save(tokenized_dataset_tensor, final_save_path)
    print(f"Processing {split} done!")

    # free memory
    del all_tokens, tokenized_dataset_tensor, tokenized_dataset


def main():
    # load fineweb-edu sample (~10B tokens)
    # data = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

    # # split into train/test
    # data = data.train_test_split(test_size=0.01, shuffle=False)

    # train_data = data["train"]
    # test_data = data["test"]  # ~100M tokens

    # shard train_data to get ~1B tokens (first 10B/2)
    # comment or uncomment as you see fit
    # train_data = train_data.shard(num_shards=2, index=0)

    ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")
    # Suggested default splits (data is pre-shuffled):
    #   rows 0..9,999       -> test  (10K stories)
    #   rows 10,000..19,999 -> val   (10K stories)
    #   rows 20,000..end    -> train (2,712,634 stories)
    test  = ds.select(range(0, 10_000))
    # val   = ds.select(range(10_000, 20_000))
    # train = ds.select(range(20_000, len(ds)))
    train = ds.select(range(20_000, 120_000))

    # tokenize and save both splits
    tokenize_and_save("train", train, save_path)
    tokenize_and_save("test", test, save_path)

    print("Done!")


if __name__ == "__main__":
    main()
