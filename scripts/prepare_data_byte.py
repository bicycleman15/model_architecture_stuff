"""
Tokenize into byte tokens.

Usage:
    python prepare_data_byte.py

"""

import os
import numpy as np
import torch
from itertools import chain

from datasets import load_dataset

from byte_tokenizer import ByteTokenizer
tokenizer = ByteTokenizer()

save_path = "data/fineweb-1b-byte"


def preprocess_and_tokenize_batched(examples):
    """
    Batched tokenization (flash-linear-attention style).
    - Processes multiple documents at once
    - Chains all tokens together efficiently
    - Pattern: [BOS] doc1 [BOS] doc2 [BOS] doc3 ...
    """
    texts = examples["text"]
    results = tokenizer.encode(texts, add_bos=True)
    input_ids = [r["input_ids"].tolist() for r in results]
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
    all_tokens = np.fromiter(chain(*tokenized_dataset["input_ids"]), dtype=np.uint8)
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
    data = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

    # # split into train/test
    data = data.train_test_split(test_size=0.01, shuffle=False)

    train = data["train"]
    test = data["test"]  # ~100M tokens

    # shard train to get ~200M gpt2 tokens i.e. ~1B byte tokens
    train = train.shard(num_shards=50, index=0)

    # ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")
    # Suggested default splits (data is pre-shuffled):
    #   rows 0..9,999       -> test  (10K stories)
    #   rows 10,000..19,999 -> val   (10K stories)
    #   rows 20,000..end    -> train (2,712,634 stories)
    # test  = ds.select(range(0, 10_000))
    # val   = ds.select(range(10_000, 20_000))
    # train = ds.select(range(20_000, len(ds)))
    # train = ds.select(range(20_000, 700_000))

    # tokenize and save both splits
    tokenize_and_save("train", train, save_path)
    tokenize_and_save("test", test, save_path)

    print("Done!")


if __name__ == "__main__":
    main()
