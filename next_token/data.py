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

Vocab size = `num_nodes + 8` covering `0..num_nodes-1`, '|', '=', '/', '$',
plus four CoT tags '<think>', '</think>', '<backtrack>', '<eos>' used by the
CoT pre-training generator (`star_graph_cot`). The '<eos>' tag is appended
to every CoT trace so the model can learn when to stop generating.

The '$' token is reserved as a teacherless dummy (Bachmann & Nagarajan §6).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Star-graph generation
# -----------------------------------------------------------------------------


def _sample_star_graph(
    deg: int,
    path_len: int,
    num_nodes: int,
    rng: np.random.Generator,
):
    """Vectorized one-shot sampling of a star-graph instance.

    Draws all ``path_len + (deg-1)*(path_len-1)`` unique nodes in a single
    ``rng.choice(replace=False)`` call (no rejection loops). Returns the
    main path, shuffled edge list, source/goal, and the list of decoy
    chains (each of length ``path_len-1``, *excluding* the shared source).
    """
    n_needed = path_len + (deg - 1) * (path_len - 1)
    if n_needed > num_nodes:
        raise ValueError(
            f"need {n_needed} unique nodes for deg={deg} path_len={path_len}, "
            f"but num_nodes={num_nodes}"
        )

    nodes = rng.choice(num_nodes, size=n_needed, replace=False).tolist()
    path = nodes[:path_len]
    source, goal = path[0], path[-1]

    chain_len = path_len - 1
    decoys: list[list[int]] = []
    idx = path_len
    for _ in range(deg - 1):
        decoys.append(nodes[idx : idx + chain_len])
        idx += chain_len

    edge_list: list[list[int]] = []
    for i in range(path_len - 1):
        edge_list.append([path[i], path[i + 1]])
    for decoy in decoys:
        edge_list.append([source, decoy[0]])
        for i in range(len(decoy) - 1):
            edge_list.append([decoy[i], decoy[i + 1]])

    rng.shuffle(edge_list)
    return path, edge_list, source, goal, decoys


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

    path, edge_list, source, goal, _ = _sample_star_graph(deg, path_len, num_nodes, rng)
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
    backtrack_weights: list[float] | None = None,
    rng: np.random.Generator | None = None,
):
    """Sample one CoT-with-backtracking star-graph instance.

    Trace shape (token ids returned in `trace_ids`):
        [source, *decoy_1[:d_1], source, *decoy_2[:d_2], ...,
                  source, *decoy_N[:d_N], *correct_path, <eos>]

    where ``N = n_back`` is sampled over the closed range
    ``[min_backtracks, max_backtracks]``: uniformly when
    ``backtrack_weights is None`` (default), or with explicit per-bucket
    weights ``backtrack_weights`` (any non-negative values; auto-normalised
    to a probability distribution; must have length
    ``max_backtracks - min_backtracks + 1``). The ``n_back`` decoys are
    sampled uniformly without replacement from the ``deg - 1`` decoys, and
    per-decoy depths are ``d_i ~ Uniform{min_depth..max_depth}``. There are
    no explicit ``<think>`` / ``</think>`` / ``<backtrack>`` markers: every
    chain starts by re-emitting ``source``, so a repeated ``source`` token
    implicitly encodes "dead end, restarting". The correct path itself
    starts with ``source``, ends at ``goal``, and is followed by ``<eos>``
    -- so the last ``path_len`` tokens before ``<eos>`` are always the
    correct path.

    Defaults give "various levels": ``max_backtracks = deg - 1`` and
    ``min_depth = max_depth = path_len - 1``.

    The correct path is always emitted forward (``source -> ... -> goal``):
    reverse-encoded targets don't make sense for an explicit search trace.

    Args:
        tokenizer: must satisfy ``tokenizer.num_nodes == num_nodes``;
            only ``EOS`` is read off it for the trace.

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
    n_choices = max_backtracks - min_backtracks + 1
    if backtrack_weights is not None:
        if len(backtrack_weights) != n_choices:
            raise ValueError(
                f"backtrack_weights length {len(backtrack_weights)} != "
                f"max_backtracks - min_backtracks + 1 = {n_choices}"
            )
        w = np.asarray(backtrack_weights, dtype=np.float64)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(
                "backtrack_weights must be non-negative with positive sum, "
                f"got {list(backtrack_weights)}"
            )

    path, edge_list, source, goal, decoys = _sample_star_graph(
        deg, path_len, num_nodes, rng
    )

    if backtrack_weights is None:
        n_back = int(rng.integers(min_backtracks, max_backtracks + 1))
    else:
        w = np.asarray(backtrack_weights, dtype=np.float64)
        p = w / w.sum()
        n_back = int(rng.choice(np.arange(min_backtracks, max_backtracks + 1), p=p))
    if n_back > 0:
        chosen_idxs = rng.permutation(len(decoys))[:n_back]
        chosen_decoys = [decoys[i] for i in chosen_idxs]
        depths = [int(rng.integers(min_depth, max_depth + 1)) for _ in range(n_back)]
    else:
        chosen_decoys, depths = [], []

    trace_ids: list[int] = []
    for decoy, d in zip(chosen_decoys, depths):
        # Each exploration chain starts by re-emitting ``source`` -- this
        # repeated source is the implicit "dead end, restarting" marker
        # since there is no <backtrack> token in the new format.
        trace_ids.append(source)
        trace_ids.extend(decoy[:d])
    # ``path`` already starts with ``source`` and ends at ``goal``; the trace
    # then terminates immediately with <eos>, so the last ``path_len`` tokens
    # before <eos> are always the correct path.
    trace_ids.extend(path)
    trace_ids.append(tokenizer.EOS)

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
    ``tokenizer.decode(trace_ids)``. In the current tag-free format, the
    trace is just node ids separated by commas and terminated with ``<eos>``;
    chain boundaries are marked implicitly by repeated ``source`` tokens.
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
    and four multi-character CoT tags '<think>', '</think>', '<backtrack>',
    '<eos>'.

    Multi-digit numbers are read greedily; commas in the input are skipped
    (they are purely a visual separator in the paper's format). When the
    cursor sees '<', the encoder greedy-matches the longest CoT tag and
    raises if none matches.
    """

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.vocab_size = num_nodes + 8

        self.PIPE = num_nodes
        self.EQ = num_nodes + 1
        self.SLASH = num_nodes + 2
        self.DUMMY = num_nodes + 3  # '$' (teacherless dummy)
        self.THINK_OPEN = num_nodes + 4  # '<think>'
        self.THINK_CLOSE = num_nodes + 5  # '</think>'
        self.BACKTRACK = num_nodes + 6  # '<backtrack>'
        self.EOS = num_nodes + 7  # '<eos>'

        self._special_to_id = {"|": self.PIPE, "=": self.EQ, "/": self.SLASH, "$": self.DUMMY}

        # Multi-char tags ordered by descending length so longest-prefix-match works.
        self._multi_char_tags: list[tuple[str, int]] = [
            ("<backtrack>", self.BACKTRACK),
            ("</think>", self.THINK_CLOSE),
            ("<think>", self.THINK_OPEN),
            ("<eos>", self.EOS),
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


def _spawn_seeds(seed: int, n: int) -> list[int]:
    """Deterministic, well-distributed child seeds for parallel workers."""
    children = np.random.SeedSequence(seed).spawn(n)
    return [int(s.generate_state(1)[0]) for s in children]


def _resolve_workers(num_workers: int | None) -> int:
    if num_workers is None or num_workers <= 0:
        return max(1, (os.cpu_count() or 1))
    return int(num_workers)


def _gen_supervised_chunk(args):
    seed, n, deg, path_len, num_nodes, reverse = args
    rng = np.random.default_rng(seed)
    parts: list[str] = []
    for _ in range(n):
        path, edges, source, goal = star_graph(
            deg, path_len, num_nodes, reverse=reverse, rng=rng
        )
        parts.append(format_sample(path, edges, source, goal))
    return "\n".join(parts)


def _build_prefix_ids(
    edges: list[list[int]],
    source: int,
    goal: int,
    tokenizer: "NumeralTokenizer",
) -> list[int]:
    """Construct the tokenized prefix directly from edges/source/goal.

    Equivalent to ``tokenizer.encode(format_sample(...).split('=')[0] + '=')``
    but avoids re-tokenizing the formatted text. Output sequence:

        [a0, b0, PIPE, a1, b1, PIPE, ..., a_n, b_n, SLASH, source, goal, EQ]
    """
    out: list[int] = []
    n_edges = len(edges)
    for i, (a, b) in enumerate(edges):
        out.append(a)
        out.append(b)
        if i < n_edges - 1:
            out.append(tokenizer.PIPE)
    out.append(tokenizer.SLASH)
    out.append(source)
    out.append(goal)
    out.append(tokenizer.EQ)
    return out


def _gen_cot_chunk_bin(args):
    """Worker: returns ``(text_chunk, bin_bytes, lens)`` per task.

    * ``text_chunk`` -- newline-joined human-readable lines (for ``train.txt``).
    * ``bin_bytes``  -- ``np.uint16`` packed token ids (prefix + trace) for the
      whole chunk, ready to append to ``train.bin``.
    * ``lens``       -- ``np.int32`` per-sample full lengths (prefix + trace),
      used to build ``train.idx`` cumulative offsets in the main process.
    """
    (
        seed,
        n,
        deg,
        path_len,
        num_nodes,
        min_backtracks,
        max_backtracks,
        min_depth,
        max_depth,
        backtrack_weights,
    ) = args
    rng = np.random.default_rng(seed)
    tokenizer = NumeralTokenizer(num_nodes)
    text_parts: list[str] = []
    ids_list: list[np.ndarray] = []
    lens = np.empty(n, dtype=np.int32)
    for k in range(n):
        _, edges, source, goal, trace_ids, _, _ = star_graph_cot(
            deg, path_len, num_nodes, tokenizer,
            min_backtracks=min_backtracks,
            max_backtracks=max_backtracks,
            min_depth=min_depth,
            max_depth=max_depth,
            backtrack_weights=backtrack_weights,
            rng=rng,
        )
        prefix_ids = _build_prefix_ids(edges, source, goal, tokenizer)
        full = np.asarray(prefix_ids + trace_ids, dtype=np.uint16)
        ids_list.append(full)
        lens[k] = full.size
        text_parts.append(format_cot_sample(edges, source, goal, trace_ids, tokenizer))
    bin_bytes = (
        np.concatenate(ids_list).tobytes() if ids_list else b""
    )
    return "\n".join(text_parts), bin_bytes, lens


def _split_n(n_samples: int, n_chunks: int) -> list[int]:
    base, rem = divmod(n_samples, n_chunks)
    return [base + (1 if i < rem else 0) for i in range(n_chunks)]


def _write_in_parallel(
    out_path: Path,
    tasks: list,
    worker_fn,
    n_workers: int,
    *,
    progress: bool = True,
    desc: str = "samples",
) -> None:
    """Run ``worker_fn(task)`` over ``tasks`` and stream returned chunk text to ``out_path``.

    If ``progress`` is True, displays a tqdm bar that advances by the per-task
    sample count (task tuples must have the sample count at index 1).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(int(t[1]) for t in tasks)
    bar = tqdm(total=total, desc=desc, unit="samp", disable=not progress)
    try:
        with open(out_path, "w") as f:
            if n_workers == 1:
                for t in tasks:
                    chunk = worker_fn(t)
                    if chunk:
                        f.write(chunk)
                        f.write("\n")
                    bar.update(int(t[1]))
            else:
                # ``fork`` is much cheaper to start than ``spawn`` on linux and
                # is safe here because workers only touch numpy + the pure-python
                # tokenizer (no torch tensors, no CUDA state).
                ctx = mp.get_context("fork")
                with ctx.Pool(processes=n_workers) as pool:
                    for t, chunk in zip(tasks, pool.imap(worker_fn, tasks, chunksize=1)):
                        if chunk:
                            f.write(chunk)
                            f.write("\n")
                        bar.update(int(t[1]))
    finally:
        bar.close()


def write_samples(
    out_path: str | os.PathLike,
    n_samples: int,
    deg: int,
    path_len: int,
    num_nodes: int,
    reverse: bool,
    seed: int,
    *,
    num_workers: int | None = 1,
    chunk_size: int = 50_000,
    progress: bool = True,
    desc: str = "supervised",
) -> None:
    """Stream-write `n_samples` star-graph lines to `out_path` (one per line).

    Parallelized across ``num_workers`` processes (default: 1 -> in-process,
    matches the prior single-threaded behavior). Pass ``num_workers=0`` (or
    ``None``) to use ``os.cpu_count()`` workers; ``chunk_size`` controls the
    samples-per-task granularity. ``progress`` toggles a tqdm bar over
    samples; ``desc`` labels it.
    """
    n_workers = _resolve_workers(num_workers)
    n_chunks = max(1, (n_samples + chunk_size - 1) // chunk_size)
    sizes = _split_n(n_samples, n_chunks)
    seeds = _spawn_seeds(seed, n_chunks)
    tasks = [
        (seeds[i], sizes[i], deg, path_len, num_nodes, reverse) for i in range(n_chunks)
    ]
    _write_in_parallel(
        Path(out_path), tasks, _gen_supervised_chunk, n_workers,
        progress=progress, desc=desc,
    )


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
    backtrack_weights: list[float] | None = None,
    seed: int,
    num_workers: int | None = 0,
    chunk_size: int = 50_000,
    progress: bool = True,
    desc: str = "cot",
) -> None:
    """Stream-write `n_samples` CoT-with-backtracking lines to `out_path`.

    Per-sample randomness:
      * ``n_back`` ranges over ``[min_backtracks, max_backtracks]``
        (defaults to ``0..deg - 1``); uniform when ``backtrack_weights`` is
        ``None``, otherwise sampled with the given (auto-normalised)
        per-bucket weights.
      * Decoys are sampled uniformly without replacement.
      * Per-decoy depth ``d_i ~ Uniform{min_depth..max_depth}``
        (defaults to ``path_len - 1`` -> always full depth).

    The correct path is always emitted forward (``source -> ... -> goal``).

    Generation is parallelized across ``num_workers`` processes (default: 0 ->
    ``os.cpu_count()``; pass ``num_workers=1`` to disable). ``chunk_size``
    controls the samples-per-task granularity (smaller = lower IPC memory).

    Also writes a sibling ``meta.json`` next to ``out_path`` recording the
    resolved cot params and the worst-case ``max_target_len``::

        2 + max_backtracks * (max_depth + 2) + 2 * path_len + 1   # +1 for <eos>

    where each decoy contributes ``1`` (source) + ``max_depth`` (chain) + ``1``
    (``<backtrack>``) tokens.

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

    # Validate weights up-front so we fail fast (the workers would also
    # validate on every sample, but a clear error here is friendlier).
    n_back_buckets = max_backtracks - min_backtracks + 1
    if backtrack_weights is not None:
        if len(backtrack_weights) != n_back_buckets:
            raise ValueError(
                f"backtrack_weights length {len(backtrack_weights)} != "
                f"max_backtracks - min_backtracks + 1 = {n_back_buckets}"
            )
        w = np.asarray(backtrack_weights, dtype=np.float64)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(
                "backtrack_weights must be non-negative with positive sum, "
                f"got {list(backtrack_weights)}"
            )

    n_workers = _resolve_workers(num_workers)
    n_chunks = max(1, (n_samples + chunk_size - 1) // chunk_size)
    sizes = _split_n(n_samples, n_chunks)
    seeds = _spawn_seeds(seed, n_chunks)
    tasks = [
        (
            seeds[i], sizes[i], deg, path_len, num_nodes,
            min_backtracks, max_backtracks, min_depth, max_depth,
            backtrack_weights,
        )
        for i in range(n_chunks)
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = out_path.with_suffix(".bin")
    idx_path = out_path.with_suffix(".idx")

    all_lens = np.empty(n_samples, dtype=np.int32)
    cursor = 0
    bar = tqdm(total=n_samples, desc=desc, unit="samp", disable=not progress)
    text_f = open(out_path, "w")
    bin_f = open(bin_path, "wb")
    try:
        if n_workers == 1:
            iterator = ((t, _gen_cot_chunk_bin(t)) for t in tasks)
            for t, (text_chunk, bin_bytes, lens) in iterator:
                if text_chunk:
                    text_f.write(text_chunk)
                    text_f.write("\n")
                if bin_bytes:
                    bin_f.write(bin_bytes)
                all_lens[cursor : cursor + len(lens)] = lens
                cursor += len(lens)
                bar.update(int(t[1]))
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_workers) as pool:
                for t, (text_chunk, bin_bytes, lens) in zip(
                    tasks, pool.imap(_gen_cot_chunk_bin, tasks, chunksize=1),
                ):
                    if text_chunk:
                        text_f.write(text_chunk)
                        text_f.write("\n")
                    if bin_bytes:
                        bin_f.write(bin_bytes)
                    all_lens[cursor : cursor + len(lens)] = lens
                    cursor += len(lens)
                    bar.update(int(t[1]))
    finally:
        text_f.close()
        bin_f.close()
        bar.close()

    if cursor != n_samples:
        raise RuntimeError(
            f"workers produced {cursor} samples, expected {n_samples}"
        )

    # idx: cumulative token offsets (length N+1 of int64). Sample i lives at
    # `bin[idx[i]:idx[i+1]]`.
    idx = np.empty(n_samples + 1, dtype=np.int64)
    idx[0] = 0
    np.cumsum(all_lens, dtype=np.int64, out=idx[1:])
    idx.tofile(idx_path)

    # Tag-free format: trace = N decoy chains (each: source + up to max_depth
    # decoy tokens) + correct path (length path_len, already includes source)
    # + <eos>.
    upper_bound = max_backtracks * (max_depth + 1) + path_len + 1
    if backtrack_weights is None:
        weights_meta: list[float] | None = None
    else:
        w = np.asarray(backtrack_weights, dtype=np.float64)
        weights_meta = (w / w.sum()).round(6).tolist()
    meta = {
        "deg": deg,
        "path_len": path_len,
        "num_nodes": num_nodes,
        "cot": {
            "min_backtracks": min_backtracks,
            "max_backtracks": max_backtracks,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "backtrack_weights": weights_meta,
        },
        "max_target_len": upper_bound,
        "bin_dtype": "uint16",
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

    Loading
    -------
    Two on-disk formats are supported:

    * **Binary cache (fast)**: ``data_path.with_suffix('.bin')`` packs all
      tokens as ``uint16`` and ``data_path.with_suffix('.idx')`` stores
      cumulative token offsets (``int64``). ``__init__`` ``mmap``\\ s both,
      so init is O(1) and RAM stays near zero. Produced by
      :func:`write_cot_samples`.
    * **Legacy text (slow)**: if the bin/idx files are missing, falls back to
      reading ``data_path`` (the human-readable ``train.txt``) and
      tokenizing every line up front. Kept for back-compat only.
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

        data_path = Path(data_path)
        bin_path = data_path.with_suffix(".bin")
        idx_path = data_path.with_suffix(".idx")

        if bin_path.exists() and idx_path.exists():
            self._init_from_bin(bin_path, idx_path, n_samples)
        else:
            self._init_from_text(data_path, n_samples)

    def _init_from_bin(
        self,
        bin_path: Path,
        idx_path: Path,
        n_samples: int | None,
    ) -> None:
        self._bin_mode = True
        self._bin = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self._idx = np.memmap(idx_path, dtype=np.int64, mode="r")
        n_total = int(self._idx.size) - 1
        if n_total < 0:
            raise ValueError(f"corrupt idx file {idx_path}: size={self._idx.size}")
        if n_samples is None or n_samples >= n_total:
            self._n_samples = n_total
        else:
            self._n_samples = int(n_samples)

        if self._n_samples == 0:
            raise ValueError(f"no samples available at {bin_path}")

        # max_target_len = (full_len - prefix_len) over the active range.
        starts = self._idx[: self._n_samples]
        ends = self._idx[1 : self._n_samples + 1]
        full_lens = (ends - starts).astype(np.int64, copy=False)
        if full_lens.min() < self.prefix_len + self.path_len:
            raise ValueError(
                f"sample length < prefix_len ({self.prefix_len}) + "
                f"path_len ({self.path_len}); idx file looks malformed"
            )
        self.max_target_len = int(full_lens.max() - self.prefix_len)

    def _init_from_text(self, data_path: Path, n_samples: int | None) -> None:
        self._bin_mode = False
        pairs = _read_samples(data_path)
        if n_samples is not None:
            pairs = pairs[:n_samples]
        if len(pairs) == 0:
            raise ValueError(f"no samples in {data_path}")

        full_ids: list[torch.Tensor] = []
        max_target_len = 0
        for i, (prefix, target) in enumerate(pairs):
            p = self.tokenizer.encode(prefix)
            if len(p) != self.prefix_len:
                raise ValueError(
                    f"row {i}: expected prefix={self.prefix_len} got {len(p)}"
                )
            t = self.tokenizer.encode(target)
            if t[-self.path_len :] == [] or len(t) < self.path_len:
                raise ValueError(
                    f"row {i}: trace shorter than path_len={self.path_len}: {target!r}"
                )
            full_ids.append(torch.tensor(p + t, dtype=torch.long))
            if len(t) > max_target_len:
                max_target_len = len(t)

        self.full_ids = full_ids
        self.max_target_len = max_target_len
        self._n_samples = len(full_ids)

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._bin_mode:
            start = int(self._idx[idx])
            end = int(self._idx[idx + 1])
            # uint16 -> int64 forces a copy off the read-only mmap.
            return torch.from_numpy(self._bin[start:end].astype(np.int64))
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
