# LAMOST 光谱四分类 —— 朴素贝叶斯

用朴素贝叶斯把 LAMOST 光谱数据分成四类：GALAXY（星系）、QSO（类星体）、STAR（恒星）、UNKNOWN（未知）。

---

## 文件说明

| 文件 | 作用 |
|---|---|
| `download.py` | 下载 LAMOST 原始数据 |
| `data_processing.py` | 清洗数据，按 7:3 划分训练集和验证集 |
| `train.py` | 训练朴素贝叶斯模型 |
| `val.py` | 在验证集上评估，出图 |

---

## 运行顺序

### 1. 下载数据

```powershell
python download.py
```

下载完成后，`data/` 目录里会有原始 LAMOST 数据。

### 2. 数据预处理

```powershell
python data_processing.py
```

生成两个文件：

- `data/lamost_train.csv` —— 训练集
- `data/lamost_val.csv` —— 验证集

### 3. 训练

```powershell
python train.py --top-k 5 --max-per-class 50000 --priors 0.10,0.06,0.71,0.13
```

训练完生成：

- `data/best_naive_bayes_model.joblib` —— 模型文件
- `data/training_meta.json` —— 训练信息

### 4. 验证

```powershell
python val.py
```

验证完生成：

- `data/figures/` 里的 4 张图
- `data/validation_*.csv` —— 各类指标
- `data/validation_summary.json` —— 汇总

---

## 训练命令说明

```powershell
python train.py --top-k 5 --max-per-class 50000 --priors 0.10,0.06,0.71,0.13
```

| 参数 | 含义 |
|---|---|
| `--top-k 5` | 只用互信息最强的 5 个特征 |
| `--max-per-class 50000` | 每类最多 5 万条（否则 STAR 太多会压死其他类） |
| `--priors 0.10,0.06,0.71,0.13` | 手动设置四类先验，顺序是 GALAXY, QSO, STAR, UNKNOWN |

**为什么要设先验**：数据里 STAR 占 94%、QSO 只有 0.6%，不调整的话模型直接放弃 QSO。这组先验是折中值，能让 QSO 召回率从 0% 提到 45% 左右，总准确率大概 82%。

---


## 输出结果怎么看

验证脚本生成 4 张图（在 `data/figures/`）：

| 图 | 看什么 |
|---|---|
| `confusion_matrix.png` | 哪类被分错成哪类 |
| `class_recall.png` | 每类的召回率，尤其看 QSO 是不是 0 |
| `posterior_probability.png` | 后验概率分布，判断每类好不好认 |
| `confidence_distribution.png` | 正确/错误样本的置信度对比 |

**不要只看总准确率**，一定要看 QSO 的召回率。总准确率 90% 但 QSO 召回 0%，等于模型根本没学会识别 QSO。

---

## 常见问题


**训练卡在互信息那步**：正常 1~2 分钟，脚本已经对互信息做了 10 万条采样，不会真的卡死

**中文图乱码**：Windows 一般没事，如果乱码就装个 SimHei 字体