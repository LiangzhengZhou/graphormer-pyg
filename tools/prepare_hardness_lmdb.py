import argparse
import json
import logging
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from torch_geometric.data import Data
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def normalize(self) -> "SplitRatios":
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive value.")
        return SplitRatios(
            train=self.train / total,
            val=self.val / total,
            test=self.test / total,
        )


def resolve_cif_path(
    cif_path: str, csv_dir: Path, cif_root: Optional[Path]
) -> Path:
    path = Path(cif_path)
    if path.is_absolute():
        return path
    if cif_root is not None:
        return cif_root / path
    return csv_dir / path


def _normalize_split_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized in {"train", "val", "test"}:
        return normalized
    if normalized == "valid":
        return "val"
    raise ValueError(f"Unsupported split label: {label}")


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = {"id", "cif_path", "hardness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _validate_split_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = df["split"].map(_normalize_split_label)
    counts = df["split"].value_counts()
    for split in ("train", "val", "test"):
        if counts.get(split, 0) == 0:
            raise ValueError(f"Split column is missing '{split}' samples.")
    if df["hardness"].isnull().any():
        raise ValueError("Hardness labels contain missing values.")
    return df


def _stratified_split(
    df: pd.DataFrame,
    ratios: SplitRatios,
    seed: int,
    n_bins: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    labels = pd.qcut(
        df["hardness"],
        q=min(n_bins, df.shape[0]),
        duplicates="drop",
    )
    df["_bin"] = labels
    splits = []
    ratios = ratios.normalize()
    for _, group in df.groupby("_bin"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_total = len(idx)
        n_train = int(math.floor(n_total * ratios.train))
        n_val = int(math.floor(n_total * ratios.val))
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        splits.append((train_idx, "train"))
        splits.append((val_idx, "val"))
        splits.append((test_idx, "test"))
    split_series = pd.Series(index=df.index, dtype=str)
    for indices, split_name in splits:
        split_series.loc[indices] = split_name
    df["split"] = split_series
    df = df.drop(columns=["_bin"])
    return df


def _random_split(
    df: pd.DataFrame, ratios: SplitRatios, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ratios = ratios.normalize()
    indices = df.index.to_numpy()
    rng.shuffle(indices)
    n_total = len(indices)
    n_train = int(math.floor(n_total * ratios.train))
    n_val = int(math.floor(n_total * ratios.val))
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    df = df.copy()
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def build_splits(
    df: pd.DataFrame,
    split: Optional[Sequence[float]],
    seed: int = 42,
    stratify: bool = False,
    n_bins: int = 10,
) -> pd.DataFrame:
    _ensure_required_columns(df)
    if "split" in df.columns:
        return _validate_split_column(df)

    if split is None:
        raise ValueError("Split ratios must be provided when split column absent.")
    if len(split) != 3:
        raise ValueError("Split ratios must contain three values.")
    ratios = SplitRatios(*split)
    if stratify:
        return _stratified_split(df, ratios, seed, n_bins=n_bins)
    return _random_split(df, ratios, seed)


def _neighbors_from_structure(
    structure: Structure,
    cutoff: float,
    max_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    centers, neighbors, distances, _ = structure.get_neighbor_list(r=cutoff)
    per_atom: Dict[int, List[Tuple[int, float]]] = {}
    for center, neighbor, dist in zip(centers, neighbors, distances):
        per_atom.setdefault(int(center), []).append((int(neighbor), float(dist)))
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_dist: List[float] = []
    for center, neighbor_list in per_atom.items():
        neighbor_list.sort(key=lambda item: item[1])
        limited = neighbor_list[:max_neighbors] if max_neighbors > 0 else neighbor_list
        for neighbor, dist in limited:
            edge_src.append(center)
            edge_dst.append(neighbor)
            edge_dist.append(dist)
    if not edge_src:
        raise ValueError("No edges found after neighbor selection.")
    edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    edge_attr = np.array(edge_dist, dtype=np.float32)[:, None]
    return edge_index, edge_attr


def build_graph(
    structure: Structure,
    sample_id: str,
    hardness: float,
    cif_path: str,
    cutoff: float,
    max_neighbors: int,
    max_in_degree: int,
    max_out_degree: int,
) -> Dict[str, object]:
    atomic_numbers = np.array([site.specie.Z for site in structure], dtype=np.int64)
    pos = np.array(structure.cart_coords, dtype=np.float32)
    cell = np.array(structure.lattice.matrix, dtype=np.float32)
    pbc = np.array([True, True, True], dtype=np.bool_)
    edge_index, edge_attr = _neighbors_from_structure(
        structure, cutoff=cutoff, max_neighbors=max_neighbors
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
    return {
        "id": sample_id,
        "cif_path": cif_path,
        "atomic_numbers": atomic_numbers,
        "pos": pos,
        "cell": cell,
        "pbc": pbc,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "in_degree": graph.in_degree.numpy(),
        "out_degree": graph.out_degree.numpy(),
        "y": float(hardness),
        "meta": {
            "formula": structure.composition.reduced_formula,
            "spacegroup": structure.get_space_group_info()[0],
            "lattice": structure.lattice.abc,
        },
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
        sample_id = str(row["id"])
        cif_path = str(row["cif_path"])
        split_name = row["split"]
        resolved_path = resolve_cif_path(cif_path, csv_dir, cif_root)
        try:
            structure = Structure.from_file(resolved_path)
            graph = build_graph(
                structure=structure,
                sample_id=sample_id,
                hardness=float(row["hardness"]),
                cif_path=str(resolved_path),
                get_edges=get_edges,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
            )
            graphs_by_split[split_name].append(graph)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to process %s: %s", resolved_path, exc)
            errors.append(
                {
                    "id": sample_id,
                    "cif_path": str(resolved_path),
                    "error": str(exc),
                }
            )

    id_maps: Dict[str, Dict[str, str]] = {}
    for split_name, graphs in graphs_by_split.items():
        env = lmdb.open(
            str(lmdb_root / f"{split_name}.lmdb"),
            map_size=map_size_mb * 1024 * 1024,
            subdir=True,
            lock=False,
        )
        id_map: Dict[str, str] = {}
        records = ((str(idx), graph) for idx, graph in enumerate(graphs))
        _write_lmdb_records(env, records, id_map)
        env.sync()
        env.close()
        id_maps[split_name] = id_map
        with (lmdb_root / f"{split_name}_id_map.json").open("w") as handle:
            json.dump(id_map, handle, indent=2)

    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name]
        split_df.to_csv(out_root / f"{split_name}.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(out_root / "errors.csv", index=False)

    stats = {
        "total": {
            "requested": int(len(df)),
            "failed": int(len(errors)),
            "successful": int(len(df) - len(errors)),
        },
        "splits": {
            name: _compute_split_stats(graphs)
            for name, graphs in graphs_by_split.items()
        },
    }
    with (out_root / "stats.json").open("w") as handle:
        json.dump(stats, handle, indent=2)

    summary = {
        "out_root": str(out_root),
        "lmdb_root": str(lmdb_root),
        "stats": stats,
        "errors": errors,
        "id_maps": id_maps,
    }
    return summary

def _write_lmdb_records(
    env: lmdb.Environment,
    records: Iterable[Tuple[str, Dict[str, object]]],
    id_map: Dict[str, int],
    all_ids: List[str],
) -> None:
    with env.begin(write=True) as txn:
        for key, record in records:
            payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(key.encode("utf-8"), payload)
            sample_id = record["id"]
            id_map[sample_id] = int(key)
            all_ids.append(sample_id)
        txn.put(b"__keys__", json.dumps(all_ids).encode("utf-8"))
        txn.put(b"__id_to_index__", json.dumps(id_map).encode("utf-8"))


def _compute_split_stats(graphs: List[Dict[str, object]]) -> Dict[str, object]:
    hardness = np.array([g["y"] for g in graphs], dtype=np.float64)
    num_nodes = np.array([len(g["atomic_numbers"]) for g in graphs], dtype=np.int64)
    num_edges = np.array([g["edge_index"].shape[1] for g in graphs], dtype=np.int64)
    stats = {
        "count": int(len(graphs)),
        "hardness_mean": float(hardness.mean()) if len(hardness) else None,
        "hardness_std": float(hardness.std()) if len(hardness) else None,
        "num_nodes": {
            "min": int(num_nodes.min()) if len(num_nodes) else None,
            "max": int(num_nodes.max()) if len(num_nodes) else None,
            "mean": float(num_nodes.mean()) if len(num_nodes) else None,
        },
        "num_edges": {
            "min": int(num_edges.min()) if len(num_edges) else None,
            "max": int(num_edges.max()) if len(num_edges) else None,
            "mean": float(num_edges.mean()) if len(num_edges) else None,
        },
    }
    return stats


def build_lmdb_from_csv(
    csv_path: Path,
    out_root: Path,
    cif_root: Optional[Path] = None,
    split: Optional[Sequence[float]] = None,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    max_in_degree: Optional[int] = None,
    max_out_degree: Optional[int] = None,
    seed: int = 42,
    stratify: bool = False,
    n_bins: int = 10,
    map_size_mb: int = 4096,
    verbose: bool = False,
) -> Dict[str, object]:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    csv_path = Path(csv_path)
    out_root = Path(out_root)
    csv_dir = csv_path.parent
    if cif_root is not None:
        cif_root = Path(cif_root)

    df = pd.read_csv(csv_path)
    if "split" in df.columns and split is not None:
        LOGGER.info("CSV contains split column; ignoring --split values.")
    df = build_splits(df, split=split, seed=seed, stratify=stratify, n_bins=n_bins)
    df = df[["id", "cif_path", "hardness", "split"]]

    out_root.mkdir(parents=True, exist_ok=True)
    lmdb_path = out_root / "hardness.lmdb"

    max_in_degree = max_in_degree or max_neighbors
    max_out_degree = max_out_degree or max_neighbors

    errors: List[Dict[str, str]] = []
    graphs_by_split: Dict[str, List[Dict[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    all_graphs: List[Dict[str, object]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
        sample_id = str(row["id"])
        cif_path = str(row["cif_path"])
        split_name = row["split"]
        resolved_path = resolve_cif_path(cif_path, csv_dir, cif_root)
        try:
            structure = Structure.from_file(resolved_path)
            graph = build_graph(
                structure=structure,
                sample_id=sample_id,
                hardness=float(row["hardness"]),
                cif_path=str(resolved_path),
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                max_in_degree=max_in_degree,
                max_out_degree=max_out_degree,
            )
            graphs_by_split[split_name].append(graph)
            all_graphs.append(graph)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to process %s: %s", resolved_path, exc)
            errors.append(
                {
                    "id": sample_id,
                    "cif_path": str(resolved_path),
                    "error": str(exc),
                }
            )

    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size_mb * 1024 * 1024,
        subdir=True,
        lock=False,
    )
    id_map: Dict[str, int] = {}
    all_ids: List[str] = []
    records = ((str(idx), graph) for idx, graph in enumerate(all_graphs))
    _write_lmdb_records(env, records, id_map, all_ids)
    env.sync()
    env.close()

    splits_json = out_root / "splits.json"
    split_payload = {
        "train": df[df["split"] == "train"]["id"].astype(str).tolist(),
        "valid": df[df["split"] == "val"]["id"].astype(str).tolist(),
        "test": df[df["split"] == "test"]["id"].astype(str).tolist(),
    }
    with splits_json.open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2)

    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name]
        split_df.to_csv(out_root / f"{split_name}.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(out_root / "errors.csv", index=False)

    stats = {
        "total": {
            "requested": int(len(df)),
            "failed": int(len(errors)),
            "successful": int(len(df) - len(errors)),
        },
        "splits": {
            name: _compute_split_stats(graphs)
            for name, graphs in graphs_by_split.items()
        },
    }
    with (out_root / "stats.json").open("w") as handle:
        json.dump(stats, handle, indent=2)

    summary = {
        "out_root": str(out_root),
        "lmdb_path": str(lmdb_path),
        "stats": stats,
        "errors": errors,
        "splits_json": str(splits_json),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graph LMDB datasets from hardness CSV + CIF files."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV.")
    parser.add_argument("--out-root", required=True, help="Output root directory.")
    parser.add_argument(
        "--cif-root",
        default=None,
        help="Optional root for resolving relative cif_path values.",
    )
    parser.add_argument(
        "--cutoff", type=float, default=8.0, help="Edge cutoff radius."
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=12,
        help="Maximum neighbors per atom.",
    )
    parser.add_argument(
        "--max-in-degree",
        type=int,
        default=None,
        help="Maximum in degree for preprocessing.",
    )
    parser.add_argument(
        "--max-out-degree",
        type=int,
        default=None,
        help="Maximum out degree for preprocessing.",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios if CSV lacks split column.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify splits by hardness quantiles.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of quantile bins for stratification.",
    )
    parser.add_argument(
        "--map-size-mb",
        type=int,
        default=4096,
        help="LMDB map size in MB.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_lmdb_from_csv(
        csv_path=Path(args.csv),
        out_root=Path(args.out_root),
        cif_root=Path(args.cif_root) if args.cif_root else None,
        split=args.split,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        max_in_degree=args.max_in_degree,
        max_out_degree=args.max_out_degree,
        seed=args.seed,
        stratify=args.stratify,
        n_bins=args.n_bins,
        map_size_mb=args.map_size_mb,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
