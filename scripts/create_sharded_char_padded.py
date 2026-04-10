# Copyright (c) 2026, Jatin Prakash.
# tokenize dataset and create shards for serious pretraining
# inspired somewhat from karpathy nanogpt repo
# also supports custom byte level tokenizer

# THIS SCRIPT WILL PRINT BrokenPipeError TRACEBACKS AT THE END -- THIS IS EXPECTED.
# os._exit(0) kills the main process before worker pools are cleaned up, so the
# orphaned workers get broken-pipe errors when they try to send results back.
# All data files are written correctly before the exit.

import os
import sys
import multiprocessing as mp
import numpy as np
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm
from transformers import AutoTokenizer  # pip install transformers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ------------------------------------------
local_dir = "/gpfs/data/ranganathlab/Jatin/model_architecture_stuff/data/fineweb_byte_1b_zeropadded_mod15"
# dataset_name = "karpathy/tinystories-gpt4-clean"
# remote_name = None
dataset_name = "HuggingFaceFW/fineweb-edu"
remote_name = "sample-10BT"
shard_size = 100_000_000  # 500M tokens per shard
max_tokens = 1_000_000_000  # stop after this many train tokens (None = no limit)
max_test_tokens = 100_000_000  # stop after this many test tokens (None = no limit)
TOKENIZER_NAME = "byte"  # "gpt2", "byte", or any HF tokenizer name


if TOKENIZER_NAME == "byte":
    DTYPE = np.uint8
else:
    DTYPE = np.uint16  # if you use a tokenizer with vocab > 65535, switch to np.uint32

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset(dataset_name, name=remote_name, split="train")
fw = fw.train_test_split(test_size=0.001, shuffle=False, seed=42)

fw_test = fw["test"] # 10M test
fw = fw["train"]
fw = fw.shard(num_shards=10, index=0)  # first ~1b gpt2 tokens

print(fw)
print(fw_test)

# --- tokenizer (lazy-init per process so mp.Pool works cleanly) ---
_TOKENIZER = None

def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        if TOKENIZER_NAME == "byte":
            from byte_tokenizer import ByteTokenizer
            _TOKENIZER = ByteTokenizer()
        else:
            _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    return _TOKENIZER

def pad_text(text):
    parts = []
    for c in text:
        parts.append(c)
        if c.isascii() and c.isalpha():
            n = (ord(c.lower()) - ord('a') + 1) % 15
            parts.append('\x00' * n)
    return ''.join(parts)

def tokenize(doc):
    """Tokenize a single document: [BOS] doc_tokens"""
    tok = _get_tokenizer()
    if TOKENIZER_NAME == "byte":
        result = tok.encode([pad_text(doc["text"])], add_bos=True)[0]
        tokens_np = result["input_ids"].astype(DTYPE)
    else:
        ids = tok.encode(pad_text(doc["text"]), add_special_tokens=False)
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        tokens_np = np.asarray(([bos] if bos is not None else []) + ids, dtype=DTYPE)
        if DTYPE == np.uint16:
            assert (tokens_np < 2**16).all(), (
                "Token IDs exceed uint16 range. Use a tokenizer with vocab <= 65535 "
                "or set DTYPE=np.uint32."
            )
    return tokens_np

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# --- tokenize train split into many .npy shards ---
print(f"\nTokenizing train split ({len(fw)} documents) ...")

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # leave some CPUs free
print(f"Using {nprocs} processes to tokenize and write shards of {shard_size} tokens each.")
pool = mp.Pool(nprocs)
shard_index = 0
total_tokens = 0
all_tokens_np = np.empty((shard_size,), dtype=DTYPE)
token_count = 0
progress_bar = None

for tokens in pool.imap(tokenize, fw, chunksize=16):
    total_tokens += len(tokens)
    if max_tokens is not None and total_tokens >= max_tokens:
        print(f"\nReached token limit ({max_tokens:,}), stopping.")
        break
    if token_count + len(tokens) < shard_size:
        all_tokens_np[token_count:token_count + len(tokens)] = tokens
        token_count += len(tokens)
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        split = "train"
        filename = os.path.join(DATA_CACHE_DIR, f"shard_{split}_{shard_index:06d}")
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None
        leftover = len(tokens) - remainder
        if leftover > 0:
            all_tokens_np[0:leftover] = tokens[remainder:]
        token_count = leftover

if token_count != 0:
    split = "train"
    filename = os.path.join(DATA_CACHE_DIR, f"shard_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])

print(f"Total train tokens: {total_tokens:,} across {shard_index + 1} shards")

# --- tokenize test split into a single test.npy ---
print(f"\nTokenizing test split ({len(fw_test)} documents) ...")
pool = mp.Pool(nprocs)
test_chunks = []
test_total = 0
for tokens in tqdm(pool.imap(tokenize, fw_test, chunksize=16),
                   total=len(fw_test), desc="Test tokenization"):
    test_chunks.append(tokens)
    test_total += len(tokens)
    if max_test_tokens is not None and test_total >= max_test_tokens:
        print(f"\nReached test token limit ({max_test_tokens:,}), stopping.")
        break
test_tokens = np.concatenate(test_chunks)
test_filename = os.path.join(DATA_CACHE_DIR, "test")
write_datafile(test_filename, test_tokens)
print(f"Saved {len(test_tokens):,} test tokens to {test_filename}.npy")

os._exit(0)
