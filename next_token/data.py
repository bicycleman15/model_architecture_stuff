"""Star-graph next-token dataset.

Ports the data pipeline from Bachmann & Nagarajan (2024,
https://arxiv.org/abs/2403.06963) into this codebase. References:
- Star graph generation:  https://github.com/gregorbachmann/Next-Token-Failures/blob/main/data/graphs.py
- NumeralTokenizer:       https://github.com/gregorbachmann/Next-Token-Failures/blob/main/tokenizing/numeral_tokenizer.py

Each sample is encoded as a single string

    "<a0,b0|a1,b1|...|a_{m-1},b_{m-1}>/<source,goal>=<n0,n1,...,n_{path_len-1}>"

where the prefix lists the (shuffled) edges of a star graph rooted at
`source` with `deg` outgoing chains of length `path_len-1`, and the target
is the unique `path_len`-long path from `source` to `goal`.

Vocab size = `num_nodes + 4` covering `0..num_nodes-1`, '|', '=', '/', '$'.
The '$' token is reserved as a teacherless dummy (Bachmann & Nagarajan §6).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Star-graph generation
# -----------------------------------------------------------------------------


def star_graph(
    deg: int,
    path_len: int,
    num_nodes: int,
    reverse: bool = False,
    rng: np.random.Generator | None = None,
):
    """Sample one star-graph instance.

    Returns:
        path:      list[int] of length `path_len` (source first, goal last)
        edge_list: list[list[int]] of length `(path_len - 1) * deg`, shuffled
        source:    int
        goal:      int
    """
    if rng is None:
        rng = np.random.default_rng()

    source = int(rng.integers(0, num_nodes))
    goal = int(rng.integers(0, num_nodes))
    while goal == source:
        goal = int(rng.integers(0, num_nodes))

    path = [source]
    for _ in range(path_len - 2):
        node = int(rng.integers(0, num_nodes))
        while node in path or node == goal:
            node = int(rng.integers(0, num_nodes))
        path.append(node)
    path.append(goal)

    edge_list: list[list[int]] = []
    for i in range(len(path) - 1):
        edge_list.append([path[i], path[i + 1]])

    used = set(path)
    for _ in range(deg - 1):
        node = source
        for _ in range(path_len - 1):
            nxt = int(rng.integers(0, num_nodes))
            while nxt in used:
                nxt = int(rng.integers(0, num_nodes))
            edge_list.append([node, nxt])
            used.add(nxt)
            node = nxt

    rng.shuffle(edge_list)
    if reverse:
        path = path[::-1]

    return path, edge_list, source, goal


def format_sample(path: list[int], edge_list: list[list[int]], source: int, goal: int) -> str:
    """Render a star-graph instance to the paper's text format."""
    edge_str = "|".join(f"{a},{b}" for a, b in edge_list)
    path_str = ",".join(str(n) for n in path)
    return f"{edge_str}/{source},{goal}={path_str}"


# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------


_DIGITS = set("0123456789")


class NumeralTokenizer:
    """Per-symbol tokenizer over `0..num_nodes-1` plus '|', '=', '/', '$'.

    Multi-digit numbers are read greedily; commas in the input are skipped
    (they are purely a visual separator in the paper's format).
    """

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.vocab_size = num_nodes + 4

        self.PIPE = num_nodes
        self.EQ = num_nodes + 1
        self.SLASH = num_nodes + 2
        self.DUMMY = num_nodes + 3  # '$' (teacherless dummy)

        self._special_to_id = {"|": self.PIPE, "=": self.EQ, "/": self.SLASH, "$": self.DUMMY}
        self._id_to_str: dict[int, str] = {i: str(i) for i in range(num_nodes)}
        self._id_to_str.update({v: k for k, v in self._special_to_id.items()})

    def encode(self, s: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(s):
            c = s[i]
            if c == ",":
                i += 1
                continue
            if c in _DIGITS:
                j = i
                while j < len(s) and s[j] in _DIGITS:
                    j += 1
                num = int(s[i:j])
                if num >= self.num_nodes:
                    raise ValueError(f"node id {num} >= num_nodes {self.num_nodes}")
                out.append(num)
                i = j
            elif c in self._special_to_id:
                out.append(self._special_to_id[c])
                i += 1
            else:
                raise ValueError(f"unexpected character {c!r} at position {i} in {s!r}")
        return out

    def decode(self, ids: Iterable[int]) -> str:
        return ",".join(self._id_to_str[int(i)] for i in ids)


# -----------------------------------------------------------------------------
# Sequence-length math (closed form for fixed deg/path_len/num_nodes)
# -----------------------------------------------------------------------------


def compute_lengths(deg: int, path_len: int, num_nodes: int) -> tuple[int, int]:
    """Return (prefix_len, target_len) for the paper's encoding.

    With this fixed format every sample tokenizes to the *same* length:
      prefix:  2 * (path_len - 1) * deg + (path_len - 1) * deg - 1   # nodes + '|' separators
                  + 1 (slash) + 2 (source, goal) + 1 (eq)
      target:  path_len
    """
    n_edges = (path_len - 1) * deg
    prefix_len = 2 * n_edges + (n_edges - 1) + 1 + 2 + 1
    target_len = path_len
    return prefix_len, target_len


# -----------------------------------------------------------------------------
# Dataset I/O
# -----------------------------------------------------------------------------


def write_samples(
    out_path: str | os.PathLike,
    n_samples: int,
    deg: int,
    path_len: int,
    num_nodes: int,
    reverse: bool,
    seed: int,
) -> None:
    """Stream-write `n_samples` star-graph lines to `out_path` (one per line)."""
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for _ in range(n_samples):
            path, edges, source, goal = star_graph(deg, path_len, num_nodes, reverse=reverse, rng=rng)
            f.write(format_sample(path, edges, source, goal) + "\n")


def _read_samples(path: str | os.PathLike) -> list[tuple[str, str]]:
    """Read `prefix=target` lines, splitting on the first '=' (which terminates the prefix)."""
    out: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"missing '=' in {line!r}")
            prefix, target = line.split("=", 1)
            out.append((prefix + "=", target))
    return out


# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------


class StarGraphDataset(Dataset):
    """Tokenized star-graph dataset.

    Args:
        data_path:    Path to a `.txt` file produced by `write_samples`.
        tokenizer:    NumeralTokenizer (must match the one used at generation).
        n_samples:    Optional cap (use ``None`` for the whole file).
        teacherless:  If True, replace the target *input* tokens with '$' (Bachmann
                      & Nagarajan §6). The labels are unchanged.
        eval_mode:    If True, ``__getitem__`` returns the full tokenized sequence
                      (used by free-generation evaluation).

    Each item in train mode is `(input_ids, labels)` where:
      * `input_ids` is the full sequence shifted left by one (length L-1).
      * `labels` is the ground-truth next-token sequence with the prefix
        positions masked to ``-100`` so that cross-entropy ignores them.
        This matches the workspace ``Transformer`` which calls
        ``F.cross_entropy(..., ignore_index=-100)``.
    """

    def __init__(
        self,
        data_path: str | os.PathLike,
        tokenizer: NumeralTokenizer,
        n_samples: int | None = None,
        teacherless: bool = False,
        eval_mode: bool = False,
    ):
        self.tokenizer = tokenizer
        self.teacherless = teacherless
        self.eval_mode = eval_mode

        pairs = _read_samples(data_path)
        if n_samples is not None:
            pairs = pairs[:n_samples]
        if len(pairs) == 0:
            raise ValueError(f"no samples in {data_path}")

        # Tokenize; assert that all rows have the same length so we can stack.
        prefix_ids_0 = tokenizer.encode(pairs[0][0])
        target_ids_0 = tokenizer.encode(pairs[0][1])
        prefix_len = len(prefix_ids_0)
        target_len = len(target_ids_0)
        seq_len = prefix_len + target_len

        rows = torch.empty((len(pairs), seq_len), dtype=torch.long)
        rows[0] = torch.tensor(prefix_ids_0 + target_ids_0, dtype=torch.long)
        for i, (prefix, target) in enumerate(pairs[1:], start=1):
            p = tokenizer.encode(prefix)
            t = tokenizer.encode(target)
            if len(p) != prefix_len or len(t) != target_len:
                raise ValueError(
                    f"row {i}: expected prefix={prefix_len} target={target_len}, "
                    f"got prefix={len(p)} target={len(t)}"
                )
            rows[i] = torch.tensor(p + t, dtype=torch.long)

        self.tokens = rows
        self.prefix_len = prefix_len
        self.target_len = target_len
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int):
        seq = self.tokens[idx]
        if self.eval_mode:
            return seq.clone()

        # Standard next-token training: input = seq[:-1], target = seq[1:]
        input_ids = seq[:-1].clone()
        labels = seq[1:].clone()

        # Mask all prefix positions: position i predicts token i+1, so the
        # first label that contributes to the loss is at position prefix_len-1.
        labels[: self.prefix_len - 1] = -100

        if self.teacherless:
            input_ids[self.prefix_len :] = self.tokenizer.DUMMY

        return input_ids, labels

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False
