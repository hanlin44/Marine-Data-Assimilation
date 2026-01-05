# MarineDLAssimilation：海洋“背景 + 稀疏观测 → 分析场”单步重建（本科毕设工程模板）

本工程面向 **GLORYS12V1/CMEMS** 类 NetCDF 再分析数据（按天一个 `.nc`），实现同化风格的**单步重建**：

- 输入：背景场 `Xb`（由真值/再分析场生成的“退化背景”） + 稀疏观测 `Y`（由真值随机稀疏采样得到） + 掩膜 `M`
- 输出：分析场 `Xa`（网络预测的重建场）
- 方法（6组对比）：
- **cnn**：CNN 基线
- **unet**：U-Net 基线
- **lstm**：LSTM（方案A：patch 展平后喂给 LSTM；当前默认 T=1，未来可扩展到多日序列）
- **pinn**：PINNs（在损失函数加入物理残差约束；网络骨干可选 unet/cnn）
- **cnn_lstm_pinn**：CNN-LSTM-PINNs（CNN 编码 + LSTM + 物理残差）
- **unet_lstm_pinn**：U-Net-LSTM-PINNs（UNet 瓶颈 LSTM + 物理残差）

> 说明：你导师给的数据是“再分析场”，真实业务中的观测（Argo、SST、SLA 等）与背景场（模式预报）需要另行接入。
> 在本科阶段，为了能完整跑通训练流程，本工程默认用“再分析场”模拟真值，并在训练时自动生成“背景+稀疏观测”对。

---

## 0. 目录结构

```
MarineDLAssimilation/
├── README.md
├── data/                         # 你的GLORYS月数据目录（按天一个nc），默认不随工程提供
└── src/
    ├── data_loader.py            # 数据加载 + 构造(Xb, Y, M, Xa_true)
    ├── patches.py                # patch切割/重组工具
    ├── train.py                  # 训练入口（PyCharm直接运行这个）
    ├── evaluate.py               # 评估入口
    ├── models/
    │   ├── base_model.py
    │   ├── cnn_model.py
    │   ├── unet_model.py
    │   ├── lstm_model.py
    │   ├── pinn_model.py
    │   └── cnn_lstm_model.py
    └── utils/
        ├── physics.py            # 地转 + 平流扩散物理残差（PINNs）
        └── metrics.py            # RMSE/MAE/相关系数等
```

```
MarineDLAssimilation/
├── README.md
├── data/                         # 你的GLORYS月数据目录（按天一个nc），默认不随工程提供
└── src/
    ├── data_loader.py            # 数据加载 + 构造(Xb, Y, M, Xa_true)
    ├── patches.py                # patch切割/重组工具
    ├── train.py                  # 训练入口（PyCharm直接运行这个）
    ├── evaluate.py               # 评估入口
    ├── models/
    │   ├── base_model.py
    │   ├── cnn_model.py
    │   ├── unet_model.py
    │   ├── lstm_model.py
    │   ├── pinn_model.py
    │   └── cnn_lstm_model.py
    └── utils/
        ├── physics.py            # 地转 + 平流扩散物理残差（PINNs）
        └── metrics.py            # RMSE/MAE/相关系数等
```

---

## 1. 环境与依赖

建议使用 **conda**：

```bash
conda create -n marine_dl python=3.10 -y
conda activate marine_dl
pip install -r requirements.txt
```

Windows + PyCharm：请确保 PyCharm 的 Interpreter 指向该环境。

---

## 2. 数据准备

推荐你把工程放到：

```
D:\Marine Data Assimilation\MarineDLAssimilation\
```

并把数据放到工程内：

```
D:\Marine Data Assimilation\MarineDLAssimilation\data\06\
  mercatorglorys12v1_gl12_mean_19930601_R19930602.nc
  mercatorglorys12v1_gl12_mean_19930602_R19930609.nc
  ...
```

这样在 **PyCharm 直接右键运行** `src/train.py` 时，不传参数也能默认找到 `data/06`。

本工程默认从目录扫描所有 `*.nc` 文件，并按文件名中的日期排序。

---

## 3. 训练（PyCharm 一键运行）

打开 `src/train.py`，右键 Run，或命令行：
> ✅ 说明：你既可以用命令行 `python -m src.train ...`（推荐），也可以在 PyCharm 里**直接右键运行** `src/train.py`。
> 之所以两种都可行，是因为工程在 `train.py/evaluate.py` 顶部做了“脚本运行兼容”处理，避免常见的 `attempted relative import with no known parent package` 报错。


### 3.1 CNN 基线（推荐先跑通）
```bash
python -m src.train --data_dir "data\\06" --model cnn --epochs 2
```

### 3.2 U-Net
```bash
python -m src.train --data_dir "data\\06" --model unet --epochs 2
```

### 3.3 LSTM（垂向剖面序列基线）
> LSTM 在本工程中把 **depth 方向当作序列**，对每个网格点的(θ, S, u, v)剖面进行重建。
```bash
python -m src.train --data_dir "data\\06" --model lstm --epochs 2
```

### 3.4 PINNs（在损失中加入物理残差）
```bash
python -m src.train --data_dir "data\\06" --model pinn --epochs 2   --phys_geo 1.0 --phys_advdiff 0.1
```

训练输出默认写到 `runs/YYYYMMDD_HHMMSS/`，包含：
- `checkpoint.pt`：模型权重
- `meta.json`：变量与通道映射（var_slices等）
- `norm_stats.pt`：归一化参数（mean/std）

---

## 4. 评估

```bash
python -m src.evaluate --run_dir runs/20250101_120000
```

会输出各通道的 RMSE / MAE，并保存 `metrics.json`。

---

## 5. 你关心的“全部变量 + 全部深度层”

工程支持“自动读取所有变量”，但 **直接把全部变量+50层全喂给网络** 会非常吃显存与内存（通道数可能上百甚至几百）。

因此：
- 默认 include_vars 只包含常用动力变量：`thetao, so, uo, vo, zos`（并默认只取前 `--depth_max 10` 层）
- 如果你确实要全变量、全深度，可这样跑：
```bash
python -m src.train --data_dir "data\\06" --model unet   --include_vars "ALL" --target_vars "ALL" --depth_max 50
```

---

## 6. 消融实验（Ablation）

你可以用同一个训练脚本完成消融：

- 去掉物理损失：`--phys_geo 0 --phys_advdiff 0`
- 去掉某些输入变量：`--include_vars "thetao,so,zos"`
- 不同稀疏观测比例：`--obs_ratio 0.01` / `--obs_ratio 0.1`
- 不同深度层数：`--depth_max 5/10/50`
- 只在观测点算数据项（更同化）：`--data_loss_on_obs_only 1`

---

## 7. 现实接入：如何替换“模拟观测/背景”

现在 dataset 里默认：
- `Xa_true` = 再分析场（当真值）
- `Xb` = 对 Xa_true 做平滑 + 加噪（模拟背景）
- `Y` = 对 Xa_true 随机采样（模拟稀疏观测）

你之后拿到真实观测文件时，只需要在 `src/data_loader.py` 的
`_make_background_and_obs()` 里替换生成逻辑（保持输出的张量形状不变），其余模型与训练代码不变。

---

## 8. 常见问题

- **打开nc很慢**：这是正常的（文件1.2GB）。本工程是“按patch读取”，每次只读一个小块；建议：
  - patch尽量小（64/96/128）
  - `num_workers=0`（Windows多进程读nc经常会出问题）
- **显存爆了**：减少 `--depth_max`、减少变量、减小 `--patch_size`、减小 `--batch_size`。
- **低纬度地转项不稳定**：PINNs 的地转损失在低纬（f≈0）会被放大，本工程默认对 `|f|<1e-5` 处降低权重，你也可关闭地转项。

祝你先把“月样例数据”跑通，后续再迁移到 GPU 服务器全量数据。


## 附：mini_sample / sample_small.nc 需要吗？

不需要。它们只是你前期用来“探测变量/维度”的小样本文件，**正式训练只需要 `data/06` 目录下的每日原始 nc**。
如果你想做快速调试，也可以单独新建一个目录（例如 `data/06_mini`）放这些小文件，然后用 `--data_dir data\06_mini` 跑一遍流程。
