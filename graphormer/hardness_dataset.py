import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import lmdb
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class HardnessLmdbDataset(Dataset):
    def __init__(self, lmdb_path: str, ids: Optional[List[str]] = None):
        super().__init__()
        self.lmdb_path = str(lmdb_path)
        self._env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=256,
        )
        with self._env.begin(write=False) as txn:
            keys_raw = txn.get(b"__keys__")
            if keys_raw is None:
                raise ValueError("LMDB missing __keys__ metadata.")
            self._all_ids = json.loads(keys_raw.decode("utf-8"))
            index_raw = txn.get(b"__id_to_index__")
            if index_raw is None:
                raise ValueError("LMDB missing __id_to_index__ metadata.")
            self._id_to_index = json.loads(index_raw.decode("utf-8"))

        if ids is None:
            self.ids = self._all_ids
        else:
            self.ids = ids

    def len(self) -> int:
        return len(self.ids)

    def get(self, idx: int) -> Data:
        sample_id = self.ids[idx]
        sample_index = self._id_to_index[sample_id]
        with self._env.begin(write=False) as txn:
            record = txn.get(str(sample_index).encode("utf-8"))
            if record is None:
                raise KeyError(f"Missing sample index {sample_index} in LMDB.")
            data = pickle.loads(record)

        x = torch.tensor(data["atomic_numbers"], dtype=torch.float32).unsqueeze(-1)
        pos = torch.tensor(data["pos"], dtype=torch.float32)
        cell = torch.tensor(data["cell"], dtype=torch.float32)
        pbc = torch.tensor(data["pbc"], dtype=torch.bool)
        edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(data["edge_attr"], dtype=torch.float32)
        y = torch.tensor([data["y"]], dtype=torch.float32)
        in_degree = torch.tensor(data["in_degree"], dtype=torch.long)
        out_degree = torch.tensor(data["out_degree"], dtype=torch.long)

        graph = Data(
            x=x,
            pos=pos,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
        graph.in_degree = in_degree
        graph.out_degree = out_degree
        graph.sample_id = sample_id
        graph.cif_path = data.get("cif_path")

        return graph


def load_splits(split_path: str) -> Dict[str, List[str]]:
    split_path = Path(split_path)
    if split_path.suffix.lower() == ".json":
        with split_path.open("r", encoding="utf-8") as handle:
            splits = json.load(handle)
        return splits
    raise ValueError("Only JSON split files are supported.")

