"""Star-graph next-token dataset.

Ports the data pipeline from Bachmann & Nagarajan (2024,
https://arxiv.org/abs/2403.06963) into this codebase. References:
- Star graph generation:  https://github.com/gregorbachmann/Next-Token-Failures/blob/main/data/graphs.py
- NumeralTokenizer:       https://github.com/gregorbachmann/Next-Token-Failures/blob/main/tokenizing/numeral_tokenizer.py

Each (supervised) sample is encoded as a single string

    "<a0,b0|a1,b1|...|a_{m-1},b_{m-1}>/<source,goal>=<n0,n1,...,n_{path_len-1}>"

where the prefix lists the (shuffled) edges of a star graph rooted at
`source` with `deg` outgoing chains of length `path_len-1`, and the target
is the unique `path_len`-long path from `source` to `goal`.

Vocab size = `num_nodes + 7` covering `0..num_nodes-1`, '|', '=', '/', '$',
plus three CoT tags '<think>', '</think>', '<backtrack>' used by the CoT
pre-training generator (`star_graph_cot`).

The '$' token is reserved as a teacherless dummy (Bachmann & Nagarajan §6).
"""

from __future__ import annotations

import json
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
# CoT-with-backtracking star-graph generation
# -----------------------------------------------------------------------------


def _extract_chains(edge_list: list[list[int]], source: int) -> list[list[int]]:
    """Recover the `deg` chains rooted at `source` from a shuffled edge list.

    Each chain is returned as the list of nodes *after* `source` (i.e., excluding
    the root). In a star graph every non-source node has at most one out-edge,
    so the walk from each `source` neighbor is unambiguous.
    """
    adj: dict[int, int] = {}
    source_outs: list[int] = []
    for a, b in edge_list:
        if a == source:
            source_outs.append(b)
        else:
            if a in adj:
                raise RuntimeError(f"non-source node {a} has multiple out-edges")
            adj[a] = b

    chains: list[list[int]] = []
    for first in source_outs:
        chain = [first]
        cur = first
        while cur in adj:
            cur = adj[cur]
            chain.append(cur)
        chains.append(chain)
    return chains


def star_graph_cot(
    deg: int,
    path_len: int,
    num_nodes: int,
    tokenizer: "NumeralTokenizer",
    *,
    min_backtracks: int = 0,
    max_backtracks: int | None = None,
    min_depth: int | None = None,
    max_depth: int | None = None,
    rng: np.random.Generator | None = None,
):
    """Sample one CoT-with-backtracking star-graph instance.

    Trace shape (token ids returned in `trace_ids`):
        [<think>, *decoy_1[:d_1], <backtrack>, ...,
                  *decoy_N[:d_N], <backtrack>,
                  *correct_path, </think>, *correct_path]

    where ``N = n_back ~ Uniform{min_backtracks..max_backtracks}``, the
    ``n_back`` decoys are sampled uniformly without replacement from the
    ``deg - 1`` decoys, and per-decoy depths
    ``d_i ~ Uniform{min_depth..max_depth}``.

    Defaults give "various levels": ``max_backtracks = deg - 1`` and
    ``min_depth = max_depth = path_len - 1``.

    The correct path is always emitted forward (``source -> ... -> goal``):
    reverse-encoded targets don't make sense for an explicit search trace.

    Args:
        tokenizer: must satisfy ``tokenizer.num_nodes == num_nodes``;
            ``THINK_OPEN`` / ``THINK_CLOSE`` / ``BACKTRACK`` ids are read off it.

    Returns:
        path:       forward correct path ``[source, a1, ..., goal]``, length ``path_len``.
        edge_list:  shuffled edge list, same as :func:`star_graph`.
        source:     int.
        goal:       int.
        trace_ids:  list[int] of token ids for everything after the ``=``.
        n_back:     int, realized number of backtracks for this sample.
        depths:     list[int] of length ``n_back``, realized per-decoy walk depths.
    """
    if rng is None:
        rng = np.random.default_rng()
    if max_backtracks is None:
        max_backtracks = deg - 1
    if min_depth is None:
        min_depth = path_len - 1
    if max_depth is None:
        max_depth = path_len - 1

    if not (0 <= min_backtracks <= max_backtracks <= deg - 1):
        raise ValueError(
            f"need 0 <= min_backtracks <= max_backtracks <= deg-1, "
            f"got min={min_backtracks}, max={max_backtracks}, deg-1={deg - 1}"
        )
    if not (0 <= min_depth <= max_depth <= path_len - 1):
        raise ValueError(
            f"need 0 <= min_depth <= max_depth <= path_len-1, "
            f"got min={min_depth}, max={max_depth}, path_len-1={path_len - 1}"
        )
    if tokenizer.num_nodes != num_nodes:
        raise ValueError(
            f"tokenizer.num_nodes ({tokenizer.num_nodes}) != num_nodes ({num_nodes})"
        )

    path, edge_list, source, goal = star_graph(
        deg, path_len, num_nodes, reverse=False, rng=rng
    )

    chains = _extract_chains(edge_list, source)
    if len(chains) != deg:
        raise RuntimeError(f"expected {deg} chains from source, got {len(chains)}")
    correct_first = path[1]
    decoys = [c for c in chains if c[0] != correct_first]
    if len(decoys) != deg - 1:
        raise RuntimeError(f"expected {deg - 1} decoys, got {len(decoys)}")

    n_back = int(rng.integers(min_backtracks, max_backtracks + 1))
    if n_back > 0:
        chosen_idxs = rng.permutation(len(decoys))[:n_back]
        chosen_decoys = [decoys[i] for i in chosen_idxs]
        depths = [int(rng.integers(min_depth, max_depth + 1)) for _ in range(n_back)]
    else:
        chosen_decoys, depths = [], []

    trace_ids: list[int] = [tokenizer.THINK_OPEN]
    for decoy, d in zip(chosen_decoys, depths):
        trace_ids.extend(decoy[:d])
        trace_ids.append(tokenizer.BACKTRACK)
    trace_ids.extend(path)
    trace_ids.append(tokenizer.THINK_CLOSE)
    trace_ids.extend(path)

    return path, edge_list, source, goal, trace_ids, n_back, depths


def format_cot_sample(
    edge_list: list[list[int]],
    source: int,
    goal: int,
    trace_ids: list[int],
    tokenizer: "NumeralTokenizer",
) -> str:
    """Render a CoT-with-backtracking instance to ``prefix=trace`` text format.

    The prefix encoding matches :func:`format_sample`; the trace is
    ``tokenizer.decode(trace_ids)`` and so contains the literal tag strings
    ``<think>`` / ``</think>`` / ``<backtrack>`` separated by commas.
    """
    edge_str = "|".join(f"{a},{b}" for a, b in edge_list)
    trace_str = tokenizer.decode(trace_ids)
    return f"{edge_str}/{source},{goal}={trace_str}"


# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------


_DIGITS = set("0123456789")


class NumeralTokenizer:
    """Per-symbol tokenizer over `0..num_nodes-1` plus '|', '=', '/', '$',
    and three multi-character CoT tags '<think>', '</think>', '<backtrack>'.

    Multi-digit numbers are read greedily; commas in the input are skipped
    (they are purely a visual separator in the paper's format). When the
    cursor sees '<', the encoder greedy-matches the longest CoT tag and
    raises if none matches.
    """

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.vocab_size = num_nodes + 7

        self.PIPE = num_nodes
        self.EQ = num_nodes + 1
        self.SLASH = num_nodes + 2
        self.DUMMY = num_nodes + 3  # '$' (teacherless dummy)
        self.THINK_OPEN = num_nodes + 4  # '<think>'
        self.THINK_CLOSE = num_nodes + 5  # '</think>'
        self.BACKTRACK = num_nodes + 6  # '<backtrack>'

        self._special_to_id = {"|": self.PIPE, "=": self.EQ, "/": self.SLASH, "$": self.DUMMY}

        # Multi-char tags ordered by descending length so longest-prefix-match works.
        self._multi_char_tags: list[tuple[str, int]] = [
            ("<backtrack>", self.BACKTRACK),
            ("</think>", self.THINK_CLOSE),
            ("<think>", self.THINK_OPEN),
        ]

        self._id_to_str: dict[int, str] = {i: str(i) for i in range(num_nodes)}
        self._id_to_str.update({v: k for k, v in self._special_to_id.items()})
        for tag, tid in self._multi_char_tags:
            self._id_to_str[tid] = tag

    def encode(self, s: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(s):
            c = s[i]
            if c == ",":
                i += 1
                continue
            if c == "<":
                matched = False
                for tag, tid in self._multi_char_tags:
                    if s.startswith(tag, i):
                        out.append(tid)
                        i += len(tag)
                        matched = True
                        break
                if not matched:
                    raise ValueError(f"unrecognized CoT tag at position {i} in {s!r}")
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


def write_cot_samples(
    out_path: str | os.PathLike,
    n_samples: int,
    deg: int,
    path_len: int,
    num_nodes: int,
    *,
    min_backtracks: int = 0,
    max_backtracks: int | None = None,
    min_depth: int | None = None,
    max_depth: int | None = None,
    seed: int,
) -> None:
    """Stream-write `n_samples` CoT-with-backtracking lines to `out_path`.

    Per-sample randomness:
      * ``n_back ~ Uniform{min_backtracks..max_backtracks}``
        (defaults to ``0..deg - 1``).
      * Decoys are sampled uniformly without replacement.
      * Per-decoy depth ``d_i ~ Uniform{min_depth..max_depth}``
        (defaults to ``path_len - 1`` -> always full depth).

    The correct path is always emitted forward (``source -> ... -> goal``).

    Also writes a sibling ``meta.json`` next to ``out_path`` recording the
    resolved cot params and the worst-case ``max_target_len``::

        2 + max_backtracks * (max_depth + 1) + 2 * path_len

    The bound is purely a function of the params, so calls with the same
    params from different splits (train/test) are idempotent w.r.t.
    ``meta.json``.
    """
    if max_backtracks is None:
        max_backtracks = deg - 1
    if min_depth is None:
        min_depth = path_len - 1
    if max_depth is None:
        max_depth = path_len - 1

    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = NumeralTokenizer(num_nodes)

    with open(out_path, "w") as f:
        for _ in range(n_samples):
            _, edges, source, goal, trace_ids, _, _ = star_graph_cot(
                deg, path_len, num_nodes, tokenizer,
                min_backtracks=min_backtracks,
                max_backtracks=max_backtracks,
                min_depth=min_depth,
                max_depth=max_depth,
                rng=rng,
            )
            line = format_cot_sample(edges, source, goal, trace_ids, tokenizer)
            f.write(line + "\n")

    upper_bound = 2 + max_backtracks * (max_depth + 1) + 2 * path_len
    meta = {
        "deg": deg,
        "path_len": path_len,
        "num_nodes": num_nodes,
        "cot": {
            "min_backtracks": min_backtracks,
            "max_backtracks": max_backtracks,
            "min_depth": min_depth,
            "max_depth": max_depth,
        },
        "max_target_len": upper_bound,
    }
    meta_path = out_path.parent / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


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


# -----------------------------------------------------------------------------
# CoT pre-training dataset (variable-length traces)
# -----------------------------------------------------------------------------


class StarGraphCoTDataset(Dataset):
    """Tokenized CoT-with-backtracking dataset.

    Variable-length: each row is the full token sequence ``prefix + trace``
    where ``prefix`` is fixed-length (computed by :func:`compute_lengths`) and
    ``trace`` varies with the sample's realized backtrack count / depths.

    ``__getitem__`` returns the 1D ``full_ids`` tensor; padding to a common
    length is done by :func:`cot_pad_collate`.
    """

    def __init__(
        self,
        data_path: str | os.PathLike,
        tokenizer: NumeralTokenizer,
        deg: int,
        path_len: int,
        num_nodes: int,
        n_samples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.path_len = path_len

        prefix_len, _ = compute_lengths(deg, path_len, num_nodes)
        self.prefix_len = prefix_len

        pairs = _read_samples(data_path)
        if n_samples is not None:
            pairs = pairs[:n_samples]
        if len(pairs) == 0:
            raise ValueError(f"no samples in {data_path}")

        full_ids: list[torch.Tensor] = []
        max_target_len = 0
        for i, (prefix, target) in enumerate(pairs):
            p = tokenizer.encode(prefix)
            if len(p) != prefix_len:
                raise ValueError(
                    f"row {i}: expected prefix={prefix_len} got {len(p)}"
                )
            t = tokenizer.encode(target)
            if t[-path_len:] == [] or len(t) < path_len:
                raise ValueError(
                    f"row {i}: trace shorter than path_len={path_len}: {target!r}"
                )
            full_ids.append(torch.tensor(p + t, dtype=torch.long))
            if len(t) > max_target_len:
                max_target_len = len(t)

        self.full_ids = full_ids
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.full_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.full_ids[idx]


def cot_pad_collate(
    batch: list[torch.Tensor],
    *,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length CoT sequences.

    Returns
    -------
    full_ids : (B, L_max) long
        Padded full token sequences.
    lengths  : (B,) long
        Original (un-padded) length of each row.
    """
    lengths = torch.tensor([x.size(0) for x in batch], dtype=torch.long)
    L_max = int(lengths.max().item())
    B = len(batch)

    padded = torch.full((B, L_max), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, : x.size(0)] = x
    return padded, lengths


def make_cot_train_targets(
    full_ids: torch.Tensor,
    lengths: torch.Tensor,
    prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard next-token shift + prefix/pad masking for CoT batches.

    ``full_ids`` is ``(B, L)`` from :func:`cot_pad_collate`. Returns:
      input_ids = full_ids[:, :-1]                       (B, L-1)
      labels    = full_ids[:, 1:].clone() with positions before ``prefix_len-1``
                  and pad positions set to ``-100``.
    """
    input_ids = full_ids[:, :-1].clone()
    labels = full_ids[:, 1:].clone()

    labels[:, : prefix_len - 1] = -100

    B, Lm1 = labels.shape
    pos = torch.arange(Lm1, device=labels.device).unsqueeze(0).expand(B, -1)
    pad_mask = pos >= (lengths.to(labels.device) - 1).unsqueeze(1)
    labels[pad_mask] = -100
    return input_ids, labels
