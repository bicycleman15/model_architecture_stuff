import torch
import torch.nn as nn


class Compressor:
    # compressor "implicitly" decides how to chunk up the sequence
    # it acts as a router
    pass


class Decoder:
    # decoder decodes compressed chunks back to tokens
    pass


class Processor:
    # processor processes compressed chunks
    pass


class HierarchicalModel(nn.Module):
    # wires everything together

    def __init__(self, config):
        
        self.compressor = Compressor(config)
        self.processor = Processor(config)
        self.decoder = Decoder(config)

        self.emb = nn.Embedding(config.vocab_size)
        self.vocab = nn.Linear(config.dim, config.vocab_size)
    
    def forward(self, input_ids):

        # get embs
        x = self.emb(input_ids) # [B, L, D]

        # where K < L since we compressed the sequence
        x, x_compressed, boundaries = self.compressor(x) # [B, L, D], [B, K, D], [B, K]

        # just process this compressed sequence
        x_processed = self.processor(x_compressed)

        # in decoder, we need to somehow up-project this back to original sequence length
        out = self.decoder(x_compressed, boundaries) # [B, L, D]

        # project to vocab
        logits = self.vocab(out) # [B, L, D] -> [B, L, V]

        return logits