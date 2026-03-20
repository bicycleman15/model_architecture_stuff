from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders
from transformers import PreTrainedTokenizerFast

# 1. Load dataset and collect all characters
ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")
all_text = "\n".join(ds["text"][:50000])
chars = sorted(set(all_text))

# 2. Vocab: <s>=0, then chars
vocab = {"<s>": 0}
for ch in chars:
    if ch not in vocab:
        vocab[ch] = len(vocab)

# 3. One token per character
tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token=None))
tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
tok.decoder = decoders.Fuse()
tok.post_processor = processors.TemplateProcessing(
    single="<s> $A",
    special_tokens=[("<s>", 0)],
)

# 4. Wrap
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    bos_token="<s>",
)

print(f"Vocab size: {tokenizer.vocab_size}")

# Test
enc = tokenizer("Once upon a time\nThere was a cat")
print(tokenizer.convert_ids_to_tokens(enc["input_ids"]))
# ['<s>', 'O', 'n', 'c', 'e', ' ', 'u', 'p', 'o', 'n', ' ', 'a', ' ', 't', 'i', 'm', 'e']

print(tokenizer.decode(enc["input_ids"], skip_special_tokens=True))
# "Once upon a time"

print(tokenizer.decode(enc["input_ids"]))


# Save and upload
# tokenizer.save_pretrained("tinystories-char-minimal")
tokenizer.push_to_hub("bicycleman15/tinystories-gpt4-clean-tokenizer")