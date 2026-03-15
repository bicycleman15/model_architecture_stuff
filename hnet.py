import torch
import torch.nn as nn


class TransformerBlock:
    # used as layers
    pass


class Compressor:
    # compressor "implicitly" decides how to chunk up the sequence
    # it acts as a router
    def __init__(self, config):
        # TODO: fill in rest
        pass

    def chunk_and_compress(self, x):
        # get probabilities to chunk at a particular position
        # for now, let's just use a small MLP

        # x: [B, L, D]

        # do a pointwise MLP
        probs = self.should_chunk_mlp(x) # [B, L, D] -> [B, L]

        boundaries = torch.where(probs > self.threshold, dim=-1) # [B, K]

        # now we need a way to extract tokens exactly at that position
        # this is how we have decided to "compress" for now, just picking up representations at that position
        compressed_x = x[boundaries] # [B, K, D] hopefully!

        return compressed_x, boundaries, probs

    def forward(self, x):
        # x: [B, L, D]

        # go through some layers
        for layer in self.layers:
            x = layer(x)

        compressed_x, boundaries, probs = self.chunk_and_compress(x)

        return x, compressed_x, boundaries
        

class Decoder:
    # decoder decodes compressed chunks back to tokens
    def __init__(self, config):
        pass

    def forward(self, x_processed, boundaries):
        # x_processed: [B, K, D]
        # boundaries: [B, K]

        # somehow get array of how many tokens we need to up-project to
        chunk_sizes = boundaries[:, 1:] - boundaries[:, :-1] # [B, K]

        # now for every position for x_processed[i], we need to up-project to chunk_sizes[i]
        # not sure how to do this without a for loop
        # lets do a for loop for now
        
        x = list()
        for i in range(x_processed.shape[1]):
            # let's use a linear interpolation here
            x.append(x_processed[:, i, :] @ torch.linear_interpolate(self.up_project_matrix, (dim, chunk_sizes[i] * dim)))

        # now we can concat everything
        x = torch.cat(x, dim=1) # [B, L, D]

        # now pass x through some layers :)
        for layer in self.layers:
            x = layer(x)

        return x


class Processor:
    # processor processes compressed chunks
    # nothing fancy here tbh, just some layers doing some processing of the sequence


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
        out = self.decoder(x_processed, boundaries) # [B, L, D]

        # project to vocab
        logits = self.vocab(out) # [B, L, D] -> [B, L, V]

        return logits