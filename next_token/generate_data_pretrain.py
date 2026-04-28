"""CLI to generate CoT-with-backtracking pre-training datasets for star graph.

Each line is ``prefix=trace`` where ``trace`` is::

    <think> decoy_1 <backtrack> ... decoy_N <backtrack> correct_path </think> correct_path

with ``N = n_back ~ Uniform{min_backtracks..max_backtracks}`` (defaults
``0..deg-1``) decoys sampled uniformly without replacement from the
``deg-1`` decoys, and per-decoy depths
``d_i ~ Uniform{min_depth..max_depth}`` (defaults to ``path_len-1``).
The correct path is always forward (``source -> ... -> goal``) and
appears twice: once as the final reasoning step inside
``<think>...</think>``, and once as the answer after ``</think>``.

Files are written to::

    {data_dir}/deg{deg}_path{path_len}_nodes{num_nodes}_cot_b{min_b}-{max_b}_d{min_d}-{max_d}/{train,test}.txt

with a sibling ``meta.json`` recording resolved params and worst-case
``max_target_len``.

Usage::

    # default: 0..deg-1 backtracks, full decoy depth, fresh-graph per sample
    python -m next_token.generate_data_pretrain --deg=5 --path_len=5 --num_nodes=100 \
        --n_train=2000000 --n_test=20000

    # custom range
    python -m next_token.generate_data_pretrain --deg=5 --path_len=5 --num_nodes=50 \
        --n_train=2000000 --n_test=20000 \
        --min_backtracks=1 --max_backtracks=2 --min_depth=2 --max_depth=4
"""

from __future__ import annotations

from pathlib import Path

import fire

from next_token.data import write_cot_samples


def dataset_dirname_pretrain(
    deg: int,
    path_len: int,
    num_nodes: int,
    min_backtracks: int,
    max_backtracks: int,
    min_depth: int,
    max_depth: int,
) -> str:
    base = f"deg{deg}_path{path_len}_nodes{num_nodes}"
    cot = f"_cot_b{min_backtracks}-{max_backtracks}_d{min_depth}-{max_depth}"
    return base + cot


def main(
    deg: int = 2,
    path_len: int = 5,
    num_nodes: int = 50,
    n_train: int = 200_000,
    n_test: int = 20_000,
    data_dir: str = "next_token/data",
    seed: int = 0,
    overwrite: bool = False,
    min_backtracks: int = 0,
    max_backtracks: int | None = None,
    min_depth: int | None = None,
    max_depth: int | None = None,
) -> None:
    if max_backtracks is None:
        max_backtracks = deg - 1
    if min_depth is None:
        min_depth = path_len - 1
    if max_depth is None:
        max_depth = path_len - 1

    out_dir = Path(data_dir) / dataset_dirname_pretrain(
        deg, path_len, num_nodes,
        min_backtracks, max_backtracks, min_depth, max_depth,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    test_path = out_dir / "test.txt"

    if not overwrite and train_path.exists() and test_path.exists():
        print(f"[generate_data_pretrain] {train_path} and {test_path} already exist, "
              f"skipping. Pass --overwrite to regenerate.")
        return

    print(
        f"[generate_data_pretrain] writing {n_train} train samples to {train_path} "
        f"(deg={deg}, path_len={path_len}, num_nodes={num_nodes}, "
        f"min_backtracks={min_backtracks}, max_backtracks={max_backtracks}, "
        f"min_depth={min_depth}, max_depth={max_depth})"
    )
    write_cot_samples(
        train_path, n_train, deg, path_len, num_nodes,
        min_backtracks=min_backtracks, max_backtracks=max_backtracks,
        min_depth=min_depth, max_depth=max_depth,
        seed=seed,
    )
    print(f"[generate_data_pretrain] writing {n_test} test samples to {test_path}")
    write_cot_samples(
        test_path, n_test, deg, path_len, num_nodes,
        min_backtracks=min_backtracks, max_backtracks=max_backtracks,
        min_depth=min_depth, max_depth=max_depth,
        seed=seed + 1,
    )
    print("[generate_data_pretrain] done.")


if __name__ == "__main__":
    fire.Fire(main)
