"""
============================================================
LAMOST 光谱四分类 —— 朴素贝叶斯训练脚本 v2
------------------------------------------------------------
针对极端类别不平衡（STAR占94.5%）优化：
  - 分层下采样：每类最多 N 条，训练集从730万降到~15万
  - 可调先验 priors：解决 QSO 被完全压制的问题
  - 互信息采样筛选
  - 保存模型 + 特征 + 先验信息
============================================================
"""

import os
import argparse
import time
import warnings
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


# ============================================================
# 参数
# ============================================================
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

parser = argparse.ArgumentParser(description="LAMOST 朴素贝叶斯训练 v2")
parser.add_argument("--data-dir", default=os.getenv("LAMOST_DATA", os.path.join(_HERE, "data")))
parser.add_argument("--top-k", type=int, default=8, help="互信息保留前K个特征，0=全部")
parser.add_argument("--mi-samples", type=int, default=100_000, help="互信息采样上限")
parser.add_argument("--max-per-class", type=int, default=50_000,
                    help="每类最多保留多少条（下采样），0=不下采样")
parser.add_argument("--priors", type=str, default="",
                    help="四类先验，逗号分隔，顺序 GALAXY,QSO,STAR,UNKNOWN。"
                         "留空=用数据分布。例：0.25,0.25,0.25,0.25 或 0.3,0.2,0.4,0.1")
parser.add_argument("--with-disc", action="store_true", help="是否额外训练 DiscNB")
args = parser.parse_args()

DATA_DIR   = args.data_dir
TRAIN_FILE = os.path.join(DATA_DIR, "lamost_train.csv")
MODEL_PATH = os.path.join(DATA_DIR, "best_naive_bayes_model.joblib")

TARGET  = "class"
CLASSES = ["GALAXY", "QSO", "STAR", "UNKNOWN"]

RAW_COLUMNS = [
    "snru", "snrg", "snrr", "snri", "snrz",
    "z", "z_err",
    "mag1", "mag2", "mag3", "mag4", "mag5", "mag6", "mag7",
    "gaia_g_mean_mag",
]


# ============================================================
# 工具
# ============================================================
def safe_read_csv(path, **kwargs):
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"无法识别文件编码：{path}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    # 颜色指数
    X["u_g"] = df["mag1"].astype(float) - df["mag2"].astype(float)
    X["g_r"] = df["mag2"].astype(float) - df["mag3"].astype(float)
    X["r_i"] = df["mag3"].astype(float) - df["mag4"].astype(float)
    X["i_z"] = df["mag4"].astype(float) - df["mag5"].astype(float)
    X["c56"] = df["mag5"].astype(float) - df["mag6"].astype(float)
    X["c67"] = df["mag6"].astype(float) - df["mag7"].astype(float)
    X["c14"] = df["mag1"].astype(float) - df["mag4"].astype(float)
    X["c17"] = df["mag1"].astype(float) - df["mag7"].astype(float)

    # 谱线强度
    for c in ["snru", "snrg", "snrr", "snri", "snrz"]:
        X[f"log_{c}"] = np.log1p(df[c].astype(float).clip(lower=0.0))

    # 红移
    X["z"]     = df["z"].astype(float)
    X["z_err"] = df["z_err"].astype(float)

    # Gaia
    X["gaia_g_mean_mag"] = df["gaia_g_mean_mag"].astype(float)

    return X


def tic(msg):
    print(f"\n{msg}")
    return time.time()


def toc(t0):
    print(f"    完成，耗时 {time.time() - t0:.1f} 秒")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("LAMOST 光谱四分类 —— 朴素贝叶斯训练 v2")
    print("=" * 60)

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"找不到训练集：\n{TRAIN_FILE}")

    # ---------- 1. 加载 ----------
    t0 = tic(f"[1/8] 加载训练集：{TRAIN_FILE}")
    df = safe_read_csv(
        TRAIN_FILE,
        usecols=RAW_COLUMNS + [TARGET],
        low_memory=False,
    )
    print(f"    行数：{len(df):,}")
    toc(t0)

    missing = [c for c in RAW_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"训练集缺少字段：{missing}")

    # ---------- 2. 标签规范化 ----------
    y = df[TARGET].astype(str).str.strip().str.upper()

    print("\n[2/8] 原始类别分布：")
    orig_counts = y.value_counts()
    for c in CLASSES:
        cnt = int(orig_counts.get(c, 0))
        print(f"    {c:10s}: {cnt:>10,}  ({cnt/len(y):.4%})")

    # ---------- 3. 特征工程 ----------
    t0 = tic("[3/8] 特征工程")
    X_raw = engineer_features(df)
    print(f"    构造特征维度：{X_raw.shape[1]}")

    nan_mask = X_raw.notna().all(axis=1)
    print(f"    含 NaN：{int((~nan_mask).sum()):,}，有效：{int(nan_mask.sum()):,}")

    X = X_raw.loc[nan_mask].reset_index(drop=True)
    y = y.loc[nan_mask].reset_index(drop=True)
    del X_raw, df
    toc(t0)

    # ---------- 4. 分层下采样 ----------
    if args.max_per_class > 0:
        t0 = tic(f"[4/8] 分层下采样（每类最多 {args.max_per_class:,}）")
        rng = np.random.default_rng(42)
        keep_idx = []
        for c in CLASSES:
            idx_c = np.where(y.values == c)[0]
            if len(idx_c) > args.max_per_class:
                idx_c = rng.choice(idx_c, size=args.max_per_class, replace=False)
            keep_idx.append(idx_c)
        keep_idx = np.concatenate(keep_idx)
        rng.shuffle(keep_idx)

        X = X.iloc[keep_idx].reset_index(drop=True)
        y = y.iloc[keep_idx].reset_index(drop=True)
        print(f"    下采样后训练集：{len(X):,}")
        print("    新类别分布：")
        new_counts = y.value_counts()
        for c in CLASSES:
            cnt = int(new_counts.get(c, 0))
            print(f"      {c:10s}: {cnt:>10,}  ({cnt/len(y):.4%})")
        toc(t0)
    else:
        print("\n[4/8] 跳过分层下采样（--max-per-class 0）")

    # ---------- 5. 互信息采样筛选 ----------
    t0 = tic(f"[5/8] 互信息筛选（采样上限 {args.mi_samples:,}）")
    n_mi = min(len(X), args.mi_samples)
    if len(X) > n_mi:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=n_mi, replace=False)
        X_mi, y_mi = X.values[idx], y.values[idx]
    else:
        X_mi, y_mi = X.values, y.values

    mi = mutual_info_classif(X_mi, y_mi, random_state=42, n_neighbors=5)
    mi_rank = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    print("\n    特征互信息排名：")
    for i, (name, score) in enumerate(mi_rank.items(), 1):
        mark = "  ← 保留" if (args.top_k > 0 and i <= args.top_k) else ""
        print(f"      {i:2d}. {name:20s}: {score:.5f}{mark}")

    if args.top_k > 0 and args.top_k < X.shape[1]:
        selected_features = list(mi_rank.head(args.top_k).index)
    else:
        selected_features = list(X.columns)
    X_sel = X[selected_features]
    print(f"\n    最终特征（{len(selected_features)}个）：{selected_features}")
    toc(t0)

    # ---------- 6. 先验设置 ----------
    if args.priors.strip():
        priors = np.array([float(x) for x in args.priors.split(",")])
        if len(priors) != len(CLASSES):
            raise ValueError(f"priors 必须 {len(CLASSES)} 个数，顺序：{CLASSES}")
        priors = priors / priors.sum()
        print(f"\n[6/8] 使用自定义先验：")
        for c, p in zip(CLASSES, priors):
            print(f"    P({c:10s}) = {p:.4f}")
    else:
        priors = None
        print("\n[6/8] 使用数据分布先验（可能导致少数类被压制）")

    # ---------- 7. 训练 GaussianNB ----------
    t0 = tic("[7/8] 训练 GaussianNB")
    gnb = GaussianNB(priors=priors)
    gnb.fit(X_sel.values, y.values)

    y_train_pred = gnb.predict(X_sel.values)
    y_train_pred = pd.Series(y_train_pred).astype(str).str.strip().str.upper().to_numpy()
    train_acc = accuracy_score(y.values, y_train_pred)
    print(f"    训练集准确率：{train_acc:.4%}")

    print("\n    训练集分类报告：")
    print(classification_report(
        y.values, y_train_pred,
        labels=CLASSES, digits=4, zero_division=0,
    ))
    toc(t0)

    # ---------- 可选 DiscNB ----------
    best_name, best_model, best_acc = "GaussianNB", gnb, train_acc

    if args.with_disc:
        t0 = tic("    额外训练 DiscNB")
        disc = Pipeline([
            ("bin", KBinsDiscretizer(
                n_bins=10, encode="ordinal",
                strategy="quantile", subsample=200_000, dtype=np.int32,
            )),
            ("nb", CategoricalNB(alpha=1.0)),
        ])
        try:
            disc.fit(X_sel.values, y.values)
            y_d = disc.predict(X_sel.values)
            y_d = pd.Series(y_d).astype(str).str.strip().str.upper().to_numpy()
            disc_acc = accuracy_score(y.values, y_d)
            print(f"    DiscNB 训练集准确率：{disc_acc:.4%}")
            if disc_acc > best_acc:
                best_name, best_model, best_acc = "DiscNB_10", disc, disc_acc
        except Exception as e:
            print(f"    DiscNB 训练失败：{e}")
        toc(t0)

    # ---------- 8. 保存 ----------
    t0 = tic("[8/8] 保存模型")

    save_dict = {
        "model":          best_model,
        "features":       selected_features,
        "model_name":     best_name,
        "train_accuracy": float(best_acc),
        "classes":        CLASSES,
        "priors":         priors.tolist() if priors is not None else None,
        "all_features":   list(X.columns),
        "max_per_class":  args.max_per_class,
    }
    joblib.dump(save_dict, MODEL_PATH, compress=3)

    meta_path = os.path.join(DATA_DIR, "training_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name":     best_name,
            "features":       selected_features,
            "train_accuracy": float(best_acc),
            "priors":         priors.tolist() if priors is not None else None,
            "max_per_class":  args.max_per_class,
            "mi_ranking":     mi_rank.to_dict(),
        }, f, ensure_ascii=False, indent=2)

    toc(t0)

    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"  最优模型：{best_name}")
    print(f"  训练集准确率：{best_acc:.4%}")
    print(f"  使用特征数：{len(selected_features)}")
    print(f"  模型文件：{MODEL_PATH}")


if __name__ == "__main__":
    main()