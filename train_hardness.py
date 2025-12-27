import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import global_mean_pool

from graphormer.functional import precalculate_paths
from graphormer.hardness_dataset import HardnessLmdbDataset, load_splits
from graphormer.model import Graphormer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Graphormer for hardness regression.")
    parser.add_argument("--lmdb", required=True, help="LMDB path.")
    parser.add_argument("--splits", required=True, help="Splits JSON path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints and preds.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--node-dim", type=int, default=128)
    parser.add_argument("--edge-dim", type=int, default=128)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-in-degree", type=int, default=50)
    parser.add_argument("--max-out-degree", type=int, default=50)
    parser.add_argument("--max-path-distance", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_loader(dataset: HardnessLmdbDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def attach_paths(batch, max_path_distance: int) -> None:
    _, _, node_paths_length, edge_paths_tensor, edge_paths_length = precalculate_paths(
        batch,
        max_path_distance=max_path_distance,
    )
    batch.node_paths_length = node_paths_length
    batch.edge_paths_tensor = edge_paths_tensor
    batch.edge_paths_length = edge_paths_length


def train_epoch(
    model: Graphormer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_path_distance: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        attach_paths(batch, max_path_distance)
        batch = batch.to(device)
        optimizer.zero_grad()
        output = global_mean_pool(model(batch), batch.batch)
        loss = loss_fn(output, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_items += batch.num_graphs
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: Graphormer,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_path_distance: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        attach_paths(batch, max_path_distance)
        batch = batch.to(device)
        output = global_mean_pool(model(batch), batch.batch)
        loss = loss_fn(output, batch.y)
        total_loss += loss.item() * batch.num_graphs
        total_items += batch.num_graphs
    return total_loss / max(total_items, 1)


@torch.no_grad()
def predict(
    model: Graphormer,
    loader: DataLoader,
    device: torch.device,
    max_path_distance: int,
) -> List[Tuple[str, float, float]]:
    model.eval()
    predictions: List[Tuple[str, float, float]] = []
    for batch in loader:
        attach_paths(batch, max_path_distance)
        batch = batch.to(device)
        output = global_mean_pool(model(batch), batch.batch)
        batch_ids = batch.sample_id
        y_true = batch.y.view(-1).cpu().tolist()
        y_pred = output.view(-1).cpu().tolist()
        for sample_id, true_val, pred_val in zip(batch_ids, y_true, y_pred):
            predictions.append((sample_id, true_val, pred_val))
    return predictions


def write_predictions(path: Path, rows: List[Tuple[str, float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "y_true", "y_pred"])
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.splits)
    train_ids = splits.get("train", [])
    valid_ids = splits.get("valid", [])
    test_ids = splits.get("test", [])

    train_dataset = HardnessLmdbDataset(args.lmdb, ids=train_ids)
    valid_dataset = HardnessLmdbDataset(args.lmdb, ids=valid_ids)
    test_dataset = HardnessLmdbDataset(args.lmdb, ids=test_ids)

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = make_loader(valid_dataset, args.batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False)

    model = Graphormer(
        num_layers=args.num_layers,
        input_node_dim=1,
        node_dim=args.node_dim,
        input_edge_dim=1,
        edge_dim=args.edge_dim,
        output_dim=1,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        max_in_degree=args.max_in_degree,
        max_out_degree=args.max_out_degree,
        max_path_distance=args.max_path_distance,
    )

    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.max_path_distance,
        )
        valid_loss = evaluate(
            model,
            valid_loader,
            loss_fn,
            device,
            args.max_path_distance,
        )
        print(f"Epoch {epoch + 1}/{args.epochs} - train_loss: {train_loss:.6f} valid_loss: {valid_loss:.6f}")

    torch.save(model.state_dict(), output_dir / "model.pt")

    pred_train = predict(model, train_loader, device, args.max_path_distance)
    pred_valid = predict(model, valid_loader, device, args.max_path_distance)
    pred_test = predict(model, test_loader, device, args.max_path_distance)

    write_predictions(output_dir / "pred_train.csv", pred_train)
    write_predictions(output_dir / "pred_valid.csv", pred_valid)
    write_predictions(output_dir / "pred_test.csv", pred_test)


if __name__ == "__main__":
    main()
