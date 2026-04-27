"""CLI to generate star-graph train/test datasets.

Writes one ``prefix=target`` line per sample to::

    {data_dir}/deg{deg}_path{path_len}_nodes{num_nodes}{_rev?}/{train,test}.txt

The output directory layout matches what `next_token.train` expects via Hydra
config (``data.dir`` defaults to ``next_token/data``).

Usage::

    python -m next_token.generate_data --deg=5 --path_len=5 --num_nodes=50 \
        --n_train=200000 --n_test=20000

    python -m next_token.generate_data --deg=2 --path_len=5 --num_nodes=20 \
        --n_train=20000 --n_test=2000

    # reverse-encoded targets (Bachmann & Nagarajan §5)
    python -m next_token.generate_data --deg=5 --path_len=5 --num_nodes=50 \
        --n_train=200000 --n_test=20000 --reverse=True
"""

from __future__ import annotations

from pathlib import Path

import fire

from next_token.data import write_samples


def dataset_dirname(deg: int, path_len: int, num_nodes: int, reverse: bool) -> str:
    base = f"deg{deg}_path{path_len}_nodes{num_nodes}"
    return base + ("_rev" if reverse else "")


def main(
    deg: int = 2,
    path_len: int = 5,
    num_nodes: int = 50,
    n_train: int = 200_000,
    n_test: int = 20_000,
    reverse: bool = False,
    data_dir: str = "next_token/data",
    seed: int = 0,
    overwrite: bool = False,
) -> None:
    out_dir = Path(data_dir) / dataset_dirname(deg, path_len, num_nodes, reverse)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    test_path = out_dir / "test.txt"

    if not overwrite and train_path.exists() and test_path.exists():
        print(f"[generate_data] {train_path} and {test_path} already exist, skipping. "
              f"Pass --overwrite to regenerate.")
        return

    print(f"[generate_data] writing {n_train} train samples to {train_path}")
    write_samples(train_path, n_train, deg, path_len, num_nodes, reverse, seed=seed)
    print(f"[generate_data] writing {n_test} test samples to {test_path}")
    write_samples(test_path, n_test, deg, path_len, num_nodes, reverse, seed=seed + 1)
    print("[generate_data] done.")


if __name__ == "__main__":
    fire.Fire(main)
