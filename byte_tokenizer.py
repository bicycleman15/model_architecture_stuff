import numpy as np

class ByteTokenizer:
    # taken from: https://github.com/goombalab/hnet/blob/main/hnet/utils/tokenizers.py
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8

    def encode(self, seqs: list[str], add_bos: bool = False, add_eos: bool = False, **kwargs) -> list[dict[str, np.ndarray]]:
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    # changed this slightly to display bos/eos tokens
    def decode(self, tokens: np.ndarray | list[int], **kwargs) -> str:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        parts = []
        buf = []
        for t in tokens:
            if t == self.bos_idx:
                if buf:
                    parts.append(bytearray(buf).decode("utf-8", **kwargs))
                    buf = []
                parts.append("<s>")
            elif t == self.eos_idx:
                if buf:
                    parts.append(bytearray(buf).decode("utf-8", **kwargs))
                    buf = []
                parts.append("</s>")
            else:
                buf.append(t)
        if buf:
            parts.append(bytearray(buf).decode("utf-8", **kwargs))
        return "".join(parts)