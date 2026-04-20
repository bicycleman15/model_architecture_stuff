"""Generate CSV datasets for the group sequence prediction task.

Thin wrapper around the reference generator from ``automl/DeltaProduct`` with
the esoteric variants (``*_hard``, ``*_tokens``, ``limit_to_*``,
``S5_only_swaps``) stripped out. Supports standard finite groups
(``S_n``, ``A_n``, ``Z_n``) and Cartesian products written with ``_x_``,
e.g. ``S3_x_Z2``.

Output CSV schema::

    seed,input,target
    <seed>,"<space-separated element indices>","<space-separated running products>"

Usage:

    python -m state_tracking.generate_data --group=S3 --k=512  --samples=100000
    python -m state_tracking.generate_data --group=S3 --k=2048 --samples=10000
    python -m state_tracking.generate_data --group=A5 --k=256  --samples=50000 --seed=42
"""

from __future__ import annotations

import random
from functools import reduce
from itertools import product
from pathlib import Path

import fire
import polars as pl
import pyrootutils
from abstract_algebra.finite_algebras import (
    FiniteAlgebra,
    generate_cyclic_group,
    generate_symmetric_group,
)

# Anchor the default output directory at <state_tracking>/data/. We try to
# resolve the project root via pyrootutils (matches the upstream repo), but
# fall back to this file's parent so the script works even without a
# ``.project-root`` marker.
try:
    _PROJECT_ROOT = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root"
    )
    _DEFAULT_DATA_DIR = _PROJECT_ROOT / "state_tracking" / "data"
except (FileNotFoundError, Exception):  # pragma: no cover - defensive fallback
    _DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def _generate_group(spec: tuple[str, int]) -> FiniteAlgebra:
    kind, n = spec
    if kind == "S":
        return generate_symmetric_group(n)
    if kind == "Z":
        return generate_cyclic_group(n)
    if kind == "A":
        s_n = generate_symmetric_group(n)
        a_n = s_n.commutator_subalgebra()
        a_n.name = f"A{n}"
        return a_n
    raise ValueError(f"Group kind must be one of S, Z, or A; got {kind!r}")


def _parse_group(group: str) -> FiniteAlgebra:
    """Parse strings like ``"S3"``, ``"A5"``, ``"Z6"``, ``"S3_x_Z2"``."""
    parts = group.split("_x_")
    specs: list[tuple[str, int]] = []
    for p in parts:
        if len(p) < 2 or p[0] not in "SAZ":
            raise ValueError(f"Invalid group spec {p!r} in {group!r}")
        specs.append((p[0], int(p[1:])))
    groups = [_generate_group(s) for s in specs]
    return reduce(lambda x, y: x * y, groups)


def _group_reduce(lhs: int, rhs: int, G: FiniteAlgebra) -> int:
    """Index of ``G.elements[lhs] * G.elements[rhs]``."""
    return G.elements.index(G.op(G.elements[lhs], G.elements[rhs]))


def main(
    group: str,
    k: int = 10,
    samples: int | None = None,
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    seed: int | None = None,
    overwrite: bool = False,
) -> None:
    """Generate a ``{group}={k}.csv`` dataset.

    Args:
        group: Group spec, e.g. ``S3``, ``A5``, ``Z6``, ``S3_x_Z2``.
        k: Sequence length.
        samples: Number of sequences to draw. If ``None``, enumerates the full
            Cartesian product (only feasible for small groups and short k).
        data_dir: Output directory (default: ``state_tracking/data``).
        seed: RNG seed. Randomized if not provided.
        overwrite: Overwrite an existing CSV at the target path.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{group}={k}.csv"
    if out_path.exists() and not overwrite:
        print(f"Data already exists at {out_path}. Use --overwrite to regenerate.")
        return

    if seed is None:
        seed = random.randint(0, 1_000_000)
    random.seed(seed)
    print(f"Using seed {seed}")

    G = _parse_group(group)
    num_elements = len(G.elements)
    num_unique = num_elements**k
    print(f"Group {group}: |G| = {num_elements}")

    if samples is None:
        print(f"Enumerating all {num_elements}^{k} = {num_unique} sequences (unshuffled).")
        sequences: list[tuple[int, ...]] | object = product(range(num_elements), repeat=k)
    else:
        if samples > num_unique:
            print(f"Warning: {samples} > {num_unique}; capping samples to {num_unique}.")
            samples = num_unique
        print(f"Randomly sampling {samples} sequences of length {k}.")
        seen: set[tuple[int, ...]] = set()
        while len(seen) < samples:
            seen.add(tuple(random.choices(range(num_elements), k=k)))
        sequences = list(seen)

    examples: list[dict] = []
    for seq in sequences:
        acc = 0
        outputs = [acc := _group_reduce(lhs=acc, rhs=x, G=G) for x in seq]
        examples.append(
            {
                "seed": seed,
                "input": " ".join(map(str, seq)),
                "target": " ".join(map(str, outputs)),
            }
        )

    df = pl.from_dicts(examples)
    print(f"Writing data to {out_path}")
    df.write_csv(out_path)


if __name__ == "__main__":
    fire.Fire(main)
