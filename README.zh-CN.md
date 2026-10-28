# GOEF：面向全局性的少样本图像异常检测评估框架

本仓库是论文 **《Don't Miss Forest for A Tree: The Globality-Orientated Evaluation
Framework for Few-shot Image Anomaly Detection》** 的完整项目代码，覆盖论文中
从“代表性参考图选择”到“8 个方法重评估”的全部实验流程。

> 少样本异常检测（FS-IAD）以往只随机/手工挑一张正常图当参考，这张图只是正常
> 图像空间中的一个**局部点**，用它得到的分数无法代表**全局**性能，而且不同论文
> 选的参考图不同，结果无法公平比较。GOEF 用 DINOv3 特征聚类、按簇大小加权，
> 给出保持全局性的评估指标。

## 整体流程

```
正常训练图 ──DINOv3 特征──► 聚类（K-means / K-medoids / K-center / coreset / random）
                                   │  summary.csv（代表图 + 簇大小）
                 ┌─────────────────┴──────────────────┐
                 ▼                                    ▼
        第三章 D：经验验证（表 I）            第三章 F：k-shot 中心数据集 + 权重
        μ_g 与 μ_r 的误差 ε、排名             （式 21–22，Alg. 2）
                 │                                    │
                 │                          8 个 FS-IAD 方法在 62 个中心数据集上评估
                 │                                    │
                 ▼                                    ▼
        results/validation                   results/reevaluation（表 II）
```

七个阶段对应 `scripts/` 下的脚本，全部可从中断处继续：

| 阶段 | 脚本 | 产物 |
|---|---|---|
| 1 | `01_select_representatives.sh` | `sampling/<方法>/<数据集>_k5_seed<种子>/summary.csv` |
| 2 | `02_build_centers.sh` | `centers/<k>-shot/<中心>/<数据集>/`（共 62 个） |
| 3 | `03_compute_weights.sh` | `sampling/weights/{mvtec,visa}_k<k>.csv` |
| 4 | `04_run_one_shot_sweeps.sh` | 5 个方法“每张训练图当一次 1-shot”的**全量**结果 CSV |
| 5 | `05_run_center_shots.sh` | 8 个方法在每个中心数据集上的结果 CSV |
| 6 | `06_aggregate_validation.sh` | 表 I |
| 7 | `07_aggregate_reevaluation.sh` | 表 II |

一次跑完：

```bash
export GOEF_WORKSPACE=/data/goef-work      # 采样/中心/结果的存放目录
export GOEF_DATA=/data/datasets            # MVTec-AD 与 VisA 根目录
export GOEF_DINOV3=/data/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
./scripts/install_methods.sh               # 克隆 8 个上游项目并覆盖 modifications/ 中的改动文件
./scripts/run_all.sh
```

### 表 I 的六种策略共用同一份“全量 1-shot”结果

表 I 虽然比较六种策略，但对每个检测方法**只需要跑一遍**：阶段 4 会把这个方法
在“每张正常训练图各当一次 1-shot 参考图”下的全部结果写成一张表（每个方法
12 288 条）。六种策略都是在这同一张表里按不同方式取行、加权：

| 策略 | 阶段 6 的取法 |
|---|---|
| SINGLE | 每个类别取一行固定参考图（该对比种子下 `k=1` 的随机选择） |
| RE-RS / RE-CS | 取 random / coreset 选出的 `n = 5` 行，按 `1/n` 平均 |
| GOEF-KC / GOEF-KD / GOEF-KM | 取 k-center / k-medoids / k-means 的 5 张代表图，按簇大小 `N_j / N` 加权 |

`μ_g` 是该类别**全部**参考图结果的均值，近似误差始终取**绝对值**：

```
ε = | μ_g − μ_r |
```

因此策略不会因为高估或低估而“占便宜”。阶段 1 只写 `summary.csv`（记录读哪些行、
权重多少），本身不运行任何检测方法。所以完整复现表一只需一遍阶段 4 加一次
阶段 6；更换 `n`、随机种子或策略集合重新聚合时，不需要重跑方法。

只想复现某一种设置时：

```bash
./scripts/05_run_center_shots.sh 1            # 只跑表 II 的 1-shot
ONLY=patchcore ./scripts/05_run_center_shots.sh 1
DETECTORS="patchcore" ./scripts/04_run_one_shot_sweeps.sh   # 表 I 只跑一个检测方法
```

## 代码结构

| 目录 | 内容 |
|---|---|
| `goef/` | 核心实现：特征提取、5 种选择策略、中心数据集、权重（式 21–22）、加权聚合、表 I 验证、表 II 重评估 |
| `goef/cli/` | 5 个命令行入口：`sample` / `build_centers` / `weights` / `validate` / `reevaluate` |
| `modifications/` | 8 个上游项目的全部改动文件（5 个逐参考图扫描 + 8 个逐中心评估），以及统一的 `unified_evaluation.py`；每个项目附运行命令 |
| `scripts/` | 端到端流水线与上游项目安装脚本 |
| `configs/` | 两个聚合阶段的 YAML 配置 |
| `results/` | 重新生成的表 I、表 II 结果 |
| `docs/` | 公式与代码对照、数据布局、复现说明、数值差异说明、第三方来源 |
| `tests/` | 14 个单元测试（权重、聚合、验证流程） |

## 已复现的结果

**表 I（6 种评估策略的近似误差与平均排名，2700 次对比）**

| 指标 | SINGLE | RE-RS | RE-CS | GOEF-KC | GOEF-KD | GOEF-KM |
|---|---|---|---|---|---|---|
| `ϵ̄ (×100)` | 1.617 | 0.681 | 0.778 | 0.752 | 0.543 | **0.486** |
| `r̄` | 4.48 | 3.16 | 3.50 | 3.48 | 3.00 | **2.94** |

**表 II（8 个方法 1-shot 图像级 AUROC）**

| 方法 | MVTec-AD | VisA |
|---|---|---|
| PatchCore | 83.64 | 77.07 |
| WinCLIP | 93.42 | 81.63 |
| PromptAD | 93.57 | 86.82 |
| AnomalyDINO | **96.42** | 89.60 |
| FoundAD | 94.68 | 92.01 |
| INP-Former | 95.43 | 93.08 |
| UniVAD | 96.29 | **93.62** |
| SubspaceAD | 95.68 | 91.99 |

1/2/4-shot 与四项指标的完整表格见 `docs/results.md`。

## 与论文数值的一致性

| 校验项 | 结果 |
|---|---|
| 式 (22) 权重 vs 论文流水线的 k=1…5 权重表 | 最大偏差 `5e-13` |
| 表 II vs 论文归档结果（80 行） | 最大偏差 `5e-7` |
| 表 II 排名（480 行） | 完全一致 |
| 排名频次（图 5 数据） | 完全一致 |
| 表 I 的 `ϵ̄` | 六种策略均一致到小数点后 3 位 |
| 表 I 的 `r̄` | 除 SINGLE 外完全一致 |

`SINGLE` 的 `r̄` 差异（本仓库默认口径 4.473，论文 4.48）来自旧流水线的精度不一致：
其他策略存为 6 位小数，而 SINGLE 直接保留原始精度，在并列名次上略吃亏。
`goef.cli.validate --legacy_single_precision` 可逐位复现论文中的 4.4807；
`results/validation/table_validation_paper_exact.csv` 即该口径的结果。

## 用于你自己的方法

只需让方法能读取“`train/good` 中恰好有 k 张参考图”的数据集：

```bash
./scripts/02_build_centers.sh                     # 只做一次
# 在 62 个中心数据集上运行你的方法，每个中心输出一个
# Category, Image AUROC, Image AUPR, Pixel AUROC, Pixel AUPRO 的 CSV
python -m goef.cli.reevaluate \
  --method MyMethod=/path/to/MyMethod \
  --sampling_root "$GOEF_WORKSPACE/sampling" \
  --output results/my_method
```

请复用 `modifications/common/unified_evaluation.py` 的指标实现，以保证与表 II 可比。

## 相关文档

* `docs/methodology.md` —— 论文每条公式/算法对应的函数
* `docs/pipeline.md` —— 各阶段的数据流与文件格式
* `docs/datasets.md` —— 数据集、中心数据集、权重语义（含 bottle 实例）
* `docs/results.md` —— 表 I / 表 II 完整复现与数值差异说明
* `docs/third_party.md` —— 8 个上游项目的地址、许可证、改动文件
* `modifications/README.md` —— 每个项目的改动清单与运行命令

## 许可证

`goef/`、`scripts/`、`configs/`、`tests/` 及为本论文编写的适配器使用 MIT 协议；
`modifications/**/files/` 中源自上游项目的文件保留其原有许可证（Apache-2.0、MIT、
Unlicense、CC BY-NC-SA 4.0 或无许可证），详见 `docs/third_party.md`。
