# 晶体结构硬度预测流水线

本项目扩展了一套从 CIF → 图 → LMDB → 训练 → 预测 CSV 的端到端流程，用于晶体结构到硬度的回归任务。

## 1. 数据准备

准备一个 CSV 文件，包含以下列名（必须一致）：

| 列名 | 含义 |
| --- | --- |
| `id` | 样本唯一标识 |
| `cif_path` | CIF 文件路径（相对或绝对） |
| `hardness` | 硬度标签（浮点数，单位自定但需一致） |

可选列：

| 列名 | 含义 |
| --- | --- |
| `split` | `train` / `valid` / `test` |

示例：

```csv
id,cif_path,hardness
mp-10018,./cifs/mp-10018.cif,23.4
mp-862690,./cifs/mp-862690.cif,18.9
```

如果提供 `split` 列，则所有行都必须有 `split`。

## 2. 生成 LMDB 与 split

使用脚本将 CSV + CIF 转为 LMDB：

```bash
python tools/prepare_hardness_lmdb.py \
  --input-csv /path/to/data.csv \
  --output-lmdb /path/to/hardness.lmdb \
  --cif-root /optional/root/dir \
  --split-mode auto \
  --train-ratio 0.8 --valid-ratio 0.1 --test-ratio 0.1 \
  --seed 42 \
  --cutoff 6.0 --max-neighbors 50
```

关键参数：

- `--split-mode`：
  - `from_csv`：按 CSV 的 `split` 列划分
  - `auto`：按比例随机划分
- `--cutoff` / `--max-neighbors`：构图邻域参数
- `--strict`：启用后，解析失败将直接报错退出

输出：

- `hardness.lmdb`：包含所有样本图与标签
- `splits.json`：记录 train/valid/test 的 `id` 列表
- `train.csv` / `valid.csv` / `test.csv`：split 列表（默认输出到 LMDB 同目录）
- `bad_cases.csv`：CIF 解析或构图失败样本

## 3. 训练与预测输出

训练并自动生成预测结果：

```bash
python train_hardness.py \
  --lmdb /path/to/hardness.lmdb \
  --splits /path/to/splits.json \
  --output-dir runs/exp001 \
  --epochs 50 \
  --batch-size 8
```

训练结束后，会自动输出：

- `runs/exp001/pred_train.csv`
- `runs/exp001/pred_valid.csv`
- `runs/exp001/pred_test.csv`

每个 CSV 的列名固定为：

```csv
id,y_true,y_pred
```

## 4. 常见问题排查

- CIF 解析失败：查看 `bad_cases.csv` 中的 `reason` 字段
- 邻居不足或结构异常：适当增大 `--cutoff` 或 `--max-neighbors`
- 出现 NaN：检查 CIF 结构是否异常，或训练超参是否过大

## 5. LMDB 样本 schema

每条样本记录字段：

- `id`：字符串
- `pos`：`(N,3)` 原子坐标
- `atomic_numbers`：`(N,)` 原子序数
- `cell`：`(3,3)` 晶胞矩阵
- `pbc`：`(3,)` 周期性标记
- `edge_index`：`(2,E)` 边索引
- `edge_attr`：`(E,1)` 距离特征
- `y`：硬度标量
- `cif_path`：可选追溯路径
- `in_degree` / `out_degree`：预计算度数

元信息存放于 LMDB `__meta__`：

- `cutoff`、`max_neighbors`、`split_mode`、`seed`、划分比例等
