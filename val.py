"""
============================================================
LAMOST 光谱四分类 —— 朴素贝叶斯验证脚本
------------------------------------------------------------
功能：
  1. 加载 train.py 保存的模型和特征名
  2. 在 lamost_val.csv 上评估
  3. 输出后验概率（课件要求：量化分类不确定性）
  4. 生成 4 张图：混淆矩阵 / 各类召回率 / 后验概率 / 置信度分布
============================================================
"""

import os
import argparse
import warnings
import json
import time

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ============================================================
# 参数
# ============================================================
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

parser = argparse.ArgumentParser(description="LAMOST 朴素贝叶斯验证")
parser.add_argument("--data-dir", default=os.getenv("LAMOST_DATA", os.path.join(_HERE, "data")))
parser.add_argument("--val-file", default=None, help="验证集路径，默认 <data-dir>/lamost_val.csv")
args = parser.parse_args()

DATA_DIR   = args.data_dir
VAL_FILE   = args.val_file or os.path.join(DATA_DIR, "lamost_val.csv")
MODEL_PATH = os.path.join(DATA_DIR, "best_naive_bayes_model.joblib")
FIGURE_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

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
    """自动尝试多种编码"""
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"无法识别文件编码：{path}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """必须与 train.py 完全一致"""
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

    # 谱线强度代理：log(snr)
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
    print("LAMOST 光谱四分类 —— 朴素贝叶斯验证")
    print("=" * 60)

    for f in (VAL_FILE, MODEL_PATH):
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到文件：\n{f}")

    # ---------- 1. 加载模型 ----------
    t0 = tic("[1/8] 加载模型")
    saved = joblib.load(MODEL_PATH)
    model             = saved["model"]
    selected_features = saved["features"]
    model_name        = saved.get("model_name", "Unknown")
    train_acc         = saved.get("train_accuracy", None)
    print(f"    模型类型：{model_name}")
    print(f"    特征数：{len(selected_features)}")
    print(f"    特征列表：{selected_features}")
    if train_acc is not None:
        print(f"    训练集准确率：{train_acc:.4%}")
    toc(t0)

    # ---------- 2. 加载验证集 ----------
    t0 = tic(f"[2/8] 加载验证集：{VAL_FILE}")
    df = safe_read_csv(
        VAL_FILE,
        usecols=RAW_COLUMNS + [TARGET],
        low_memory=False,
    )
    print(f"    原始验证集行数：{len(df):,}")
    toc(t0)

    missing = [c for c in RAW_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"验证集缺少字段：{missing}")

    # ---------- 3. 标签规范化 ----------
    y = df[TARGET].astype(str).str.strip().str.upper()
    unknown = set(y.unique()) - set(CLASSES)
    if unknown:
        print(f"    [警告] 验证集出现未定义标签：{unknown}")

    # ---------- 4. 特征工程 ----------
    t0 = tic("[3/8] 特征工程（与训练脚本一致）")
    X_all = engineer_features(df)
    del df
    X_sel = X_all[selected_features]

    nan_mask = X_sel.notna().all(axis=1)
    n_nan = int((~nan_mask).sum())
    print(f"    含 NaN 样本：{n_nan:,}")
    print(f"    有效验证样本：{int(nan_mask.sum()):,}")

    X_valid = X_sel.loc[nan_mask].reset_index(drop=True)
    y_valid = y.loc[nan_mask].reset_index(drop=True)
    del X_all, X_sel
    toc(t0)

    # ---------- 5. 预测 + 后验概率 ----------
    t0 = tic("[4/8] 预测 + 后验概率")
    y_pred_raw = model.predict(X_valid.values)
    y_pred = pd.Series(y_pred_raw).astype(str).str.strip().str.upper().to_numpy()

    bad = set(np.unique(y_pred)) - set(CLASSES)
    if bad:
        raise ValueError(
            f"模型输出了未定义类别：{bad}\n"
            f"model.classes_ = {list(model.classes_)}"
        )

    # 课件要求：输出后验概率，量化分类不确定性
    y_prob = model.predict_proba(X_valid.values)
    confidence = y_prob.max(axis=1)
    toc(t0)

    # ---------- 6. 指标计算 ----------
    acc = accuracy_score(y_valid.values, y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_valid.values, y_pred,
        labels=CLASSES, zero_division=0,
    )
    macro_f1    = f.mean()
    weighted_f1 = np.average(f, weights=s) if s.sum() > 0 else 0.0

    print("\n" + "=" * 60)
    print("[5/8] 验证结果")
    print("=" * 60)
    print(f"    模型：{model_name}")
    print(f"    原始验证集：{len(y):,}")
    print(f"    跳过 NaN  ：{n_nan:,}")
    print(f"    实际验证  ：{len(y_valid):,}")
    print(f"    总准确率  ：{acc:.4%}")
    print(f"    Macro-F1  ：{macro_f1:.4f}")
    print(f"    Weighted-F1：{weighted_f1:.4f}")

    print("\n分类报告：")
    print(classification_report(
        y_valid.values, y_pred,
        labels=CLASSES, digits=4, zero_division=0,
    ))

    cm = confusion_matrix(y_valid.values, y_pred, labels=CLASSES)
    cm_df = pd.DataFrame(
        cm,
        index=[f"真实_{c}" for c in CLASSES],
        columns=[f"预测_{c}" for c in CLASSES],
    )
    print("混淆矩阵：")
    print(cm_df)

    metric_df = pd.DataFrame({
        "class":     CLASSES,
        "precision": p,
        "recall":    r,
        "f1":        f,
        "support":   s,
    })
    print("\n各类别指标：")
    print(metric_df.to_string(index=False))

    # ---------- 置信度统计 ----------
    correct_mask = y_valid.values == y_pred
    wrong_mask   = ~correct_mask
    print(f"\n置信度统计：")
    if correct_mask.sum() > 0:
        print(f"    正确样本平均置信度：{confidence[correct_mask].mean():.4f}")
    if wrong_mask.sum() > 0:
        print(f"    错误样本平均置信度：{confidence[wrong_mask].mean():.4f}")

    # ============================================================
    # 生成图表
    # ============================================================
    print("\n[6/8] 生成图表...")

    # ---- 图 1：混淆矩阵 ----
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(
        f"LAMOST 朴素贝叶斯 混淆矩阵  ({model_name})\n"
        f"总准确率 = {acc:.2%}",
        fontsize=13,
    )
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_yticklabels(CLASSES)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path1 = os.path.join(FIGURE_DIR, "confusion_matrix.png")
    fig.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [图1] {path1}")

    # ---- 图 2：各类别召回率 ----
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(CLASSES))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(x, r, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("类别")
    ax.set_ylabel("召回率 (Recall)")
    ax.set_title(f"LAMOST 各类别召回率  ({model_name})")
    for bar, val, cnt in zip(bars, r, s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.2%}\n(n={cnt:,})",
            ha="center", fontsize=10,
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(FIGURE_DIR, "class_recall.png")
    fig.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [图2] {path2}")

    # ---- 图 3：后验概率分布（课件核心：不确定性量化）----
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.18
    x = np.arange(len(CLASSES))
    for i, true_class in enumerate(CLASSES):
        mask = y_valid.values == true_class
        if mask.sum() == 0:
            continue
        mean_probs = y_prob[mask].mean(axis=0)
        ax.bar(
            x + (i - 1.5) * width,
            mean_probs,
            width=width,
            label=f"真实={true_class} (n={mask.sum():,})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("平均后验概率")
    ax.set_title(
        f"后验概率分布 —— 分类不确定性量化  ({model_name})\n"
        f"对角线柱越高说明该类越容易识别"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path3 = os.path.join(FIGURE_DIR, "posterior_probability.png")
    fig.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [图3] {path3}")

    # ---- 图 4：置信度分布 ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        confidence[correct_mask], bins=40, alpha=0.65,
        label=f"预测正确 (n={correct_mask.sum():,})",
        color="#55A868",
    )
    if wrong_mask.sum() > 0:
        ax.hist(
            confidence[wrong_mask], bins=40, alpha=0.75,
            label=f"预测错误 (n={wrong_mask.sum():,})",
            color="#C44E52",
        )
    ax.set_xlabel("最大后验概率（预测置信度）")
    ax.set_ylabel("样本数量")
    ax.set_title(
        f"预测置信度分布  ({model_name})\n"
        f"错误样本集中在低置信度区说明模型知道自己不确定"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path4 = os.path.join(FIGURE_DIR, "confidence_distribution.png")
    fig.savefig(path4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [图4] {path4}")

    # ============================================================
    # 保存结果
    # ============================================================
    print("\n[7/8] 保存结果...")

    # 每个样本的后验概率
    pred_df = pd.DataFrame(y_prob, columns=[f"prob_{c}" for c in CLASSES])
    pred_df.insert(0, "true_label", y_valid.values)
    pred_df.insert(1, "pred_label", y_pred)
    pred_df["confidence"] = confidence
    pred_df["is_correct"] = (y_valid.values == y_pred)
    pred_path = os.path.join(DATA_DIR, "validation_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"    [CSV] 预测（含后验概率）：{pred_path}")

    # 各类别指标
    metric_path = os.path.join(DATA_DIR, "validation_metrics.csv")
    metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")
    print(f"    [CSV] 各类别指标：{metric_path}")

    # 混淆矩阵
    cm_path = os.path.join(DATA_DIR, "validation_confusion_matrix.csv")
    cm_df.to_csv(cm_path, encoding="utf-8-sig")
    print(f"    [CSV] 混淆矩阵：{cm_path}")

    # 误分样本
    if wrong_mask.sum() > 0:
        err_df = X_valid.loc[wrong_mask].copy()
        err_df.insert(0, "true_label", y_valid.values[wrong_mask])
        err_df.insert(1, "pred_label", y_pred[wrong_mask])
        err_df["confidence"] = confidence[wrong_mask]
        for k, c in enumerate(CLASSES):
            err_df[f"prob_{c}"] = y_prob[wrong_mask, k]
        err_path = os.path.join(DATA_DIR, "validation_misclassified.csv")
        err_df.to_csv(err_path, index=False, encoding="utf-8-sig")
        print(f"    [CSV] 误分样本（n={wrong_mask.sum():,}）：{err_path}")

    # 汇总 JSON
    summary = {
        "model_name":     model_name,
        "features":       selected_features,
        "n_val_total":    int(len(y)),
        "n_val_used":     int(len(y_valid)),
        "n_skipped_nan":  int(n_nan),
        "accuracy":       float(acc),
        "macro_f1":       float(macro_f1),
        "weighted_f1":    float(weighted_f1),
        "per_class": {
            c: {
                "precision": float(p[i]),
                "recall":    float(r[i]),
                "f1":        float(f[i]),
                "support":   int(s[i]),
            } for i, c in enumerate(CLASSES)
        },
        "mean_confidence_correct": float(confidence[correct_mask].mean()) if correct_mask.sum() > 0 else None,
        "mean_confidence_wrong":   float(confidence[wrong_mask].mean())   if wrong_mask.sum()   > 0 else None,
    }
    summary_path = os.path.join(DATA_DIR, "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"    [JSON] 汇总报告：{summary_path}")

    # ============================================================
    # 收尾
    # ============================================================
    print("\n[8/8] " + "=" * 55)
    print("验证完成")
    print("=" * 60)
    print(f"  模型：{model_name}")
    print(f"  总准确率：{acc:.4%}")
    print(f"  Macro-F1：{macro_f1:.4f}")
    print(f"  Weighted-F1：{weighted_f1:.4f}")
    print(f"\n  图表目录：{FIGURE_DIR}")
    print(f"    confusion_matrix.png")
    print(f"    class_recall.png")
    print(f"    posterior_probability.png")
    print(f"    confidence_distribution.png")
    print(f"\n  结果目录：{DATA_DIR}")
    print(f"    validation_predictions.csv")
    print(f"    validation_metrics.csv")
    print(f"    validation_confusion_matrix.csv")
    print(f"    validation_summary.json")


if __name__ == "__main__":
    main()