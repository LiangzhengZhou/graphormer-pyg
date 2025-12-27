import argparse
import csv
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
from pymatgen.core import Structure
from torch_geometric.data import Data

from graphormer.functional import precalculate_custom_attributes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare hardness LMDB from CIF CSV.")
    parser.add_argument("--input-csv", required=True, help="Input CSV with id,cif_path,hardness.")
    parser.add_argument("--output-lmdb", required=True, help="Output LMDB path.")
    parser.add_argument("--cif-root", default=None, help="Root for relative CIF paths.")
    parser.add_argument("--split-mode", choices=["from_csv", "auto"], default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max-neighbors", type=int, default=50)
    parser.add_argument("--max-in-degree", type=int, default=None)
    parser.add_argument("--max-out-degree", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail on first error.")
    parser.add_argument("--splits-json", default=None, help="Output splits JSON path.")
    parser.add_argument("--splits-dir", default=None, help="Output split CSV directory.")
    parser.add_argument("--bad-cases", default=None, help="Output bad cases CSV path.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    return rows


def validate_split_column(rows: List[Dict[str, str]]) -> bool:
    has_split = any("split" in row for row in rows)
    if not has_split:
        return False
    missing = [row for row in rows if "split" not in row or not row["split"]]
    if missing:
        raise ValueError("Split column present but missing values in some rows.")
    return True


def resolve_cif_path(cif_path: str, cif_root: Path) -> Path:
    path = Path(cif_path)
    if not path.is_absolute():
        path = cif_root / path
    return path


def build_graph_from_structure(
    structure: Structure,
    cutoff: float,
    max_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    atomic_numbers = np.array([site.specie.Z for site in structure], dtype=np.int64)
    pos = np.array(structure.cart_coords, dtype=np.float32)
    cell = np.array(structure.lattice.matrix, dtype=np.float32)
    pbc = np.array([True, True, True], dtype=np.bool_)

    centers, neighbors, _, distances = structure.get_neighbor_list(r=cutoff)
    neighbor_map: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(structure))}
    for center, neighbor, distance in zip(centers, neighbors, distances):
        neighbor_map[int(center)].append((int(neighbor), float(distance)))

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_dist: List[float] = []
    for center, entries in neighbor_map.items():
        entries_sorted = sorted(entries, key=lambda item: item[1])[:max_neighbors]
        for neighbor, distance in entries_sorted:
            edge_src.append(center)
            edge_dst.append(neighbor)
            edge_dist.append(distance)

    if not edge_src:
        raise ValueError("No edges found after neighbor selection.")

    edge_index = np.stack([edge_src, edge_dst], axis=0).astype(np.int64)
    edge_attr = np.array(edge_dist, dtype=np.float32)[:, None]
    return atomic_numbers, pos, cell, pbc, edge_index, edge_attr


def create_splits(
    rows: List[Dict[str, str]],
    split_mode: str,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Dict[str, List[str]]:
    ids = [row["id"] for row in rows]
    if split_mode == "from_csv":
        splits: Dict[str, List[str]] = {"train": [], "valid": [], "test": []}
        for row in rows:
            split = row["split"].strip().lower()
            if split not in splits:
                raise ValueError(f"Invalid split value: {split}")
            splits[split].append(row["id"])
        if not splits["train"] or not splits["valid"]:
            raise ValueError("Train/valid splits must be non-empty.")
        return splits

    if train_ratio + valid_ratio + test_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive number.")
    total = train_ratio + valid_ratio + test_ratio
    train_ratio /= total
    valid_ratio /= total
    test_ratio /= total

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    n_test = n_total - n_train - n_valid

    splits = {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test": shuffled[n_train + n_valid:],
    }
    if not splits["train"] or not splits["valid"]:
        raise ValueError("Train/valid splits must be non-empty.")
    return splits


def write_split_csvs(splits: Dict[str, List[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in splits.items():
        path = output_dir / f"{split_name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id"])
            for sample_id in ids:
                writer.writerow([sample_id])


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    cif_root = Path(args.cif_root) if args.cif_root else input_csv.parent
    output_lmdb = Path(args.output_lmdb)

    rows = read_csv_rows(input_csv)
    if not rows:
        raise ValueError("No rows found in input CSV.")

    if args.split_mode == "from_csv":
        if not validate_split_column(rows):
            raise ValueError("split-mode from_csv requires split column in CSV.")

    splits = create_splits(
        rows,
        split_mode=args.split_mode,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    max_in_degree = args.max_in_degree or args.max_neighbors
    max_out_degree = args.max_out_degree or args.max_neighbors

    output_lmdb.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_lmdb), map_size=1099511627776)

    bad_cases_path = Path(args.bad_cases) if args.bad_cases else output_lmdb.parent / "bad_cases.csv"
    bad_cases: List[List[str]] = []
    all_ids: List[str] = []
    id_to_index: Dict[str, int] = {}

    success = 0
    with env.begin(write=True) as txn:
        for idx, row in enumerate(rows):
            sample_id = row["id"]
            cif_path = resolve_cif_path(row["cif_path"], cif_root)
            hardness = float(row["hardness"])
            try:
                structure = Structure.from_file(str(cif_path))
                (
                    atomic_numbers,
                    pos,
                    cell,
                    pbc,
                    edge_index,
                    edge_attr,
                ) = build_graph_from_structure(
                    structure,
                    cutoff=args.cutoff,
                    max_neighbors=args.max_neighbors,
                )
                graph = Data(
                    x=np.zeros((len(atomic_numbers), 1), dtype=np.float32),
                    edge_index=edge_index,
                )
                graph = precalculate_custom_attributes(
                    graph,
                    max_in_degree=max_in_degree,
                    max_out_degree=max_out_degree,
                )
                record = {
                    "id": sample_id,
                    "cif_path": str(cif_path),
                    "atomic_numbers": atomic_numbers,
                    "pos": pos,
                    "cell": cell,
                    "pbc": pbc,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "in_degree": graph.in_degree.numpy(),
                    "out_degree": graph.out_degree.numpy(),
                    "y": hardness,
                    "cutoff": args.cutoff,
                    "max_neighbors": args.max_neighbors,
                }
                key = str(idx).encode("utf-8")
                txn.put(key, pickle.dumps(record))
                all_ids.append(sample_id)
                id_to_index[sample_id] = idx
                success += 1
            except Exception as exc:
                reason = str(exc)
                bad_cases.append([sample_id, str(cif_path), reason])
                if args.strict:
                    raise

        txn.put(b"__keys__", json.dumps(all_ids).encode("utf-8"))
        txn.put(b"__id_to_index__", json.dumps(id_to_index).encode("utf-8"))
        txn.put(
            b"__meta__",
            json.dumps(
                {
                    "cutoff": args.cutoff,
                    "max_neighbors": args.max_neighbors,
                    "split_mode": args.split_mode,
                    "seed": args.seed,
                    "train_ratio": args.train_ratio,
                    "valid_ratio": args.valid_ratio,
                    "test_ratio": args.test_ratio,
                }
            ).encode("utf-8"),
        )

    if bad_cases:
        with bad_cases_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "cif_path", "reason"])
            writer.writerows(bad_cases)

    splits_json = Path(args.splits_json) if args.splits_json else output_lmdb.parent / "splits.json"
    with splits_json.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)

    if args.splits_dir:
        write_split_csvs(splits, Path(args.splits_dir))
    else:
        write_split_csvs(splits, output_lmdb.parent)

    total = len(rows)
    print(f"Total samples: {total}")
    print(f"Success: {success}")
    print(f"Failed: {len(bad_cases)}")
    for split_name, ids in splits.items():
        print(f"{split_name}: {len(ids)}")


if __name__ == "__main__":
    main()
