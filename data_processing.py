import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ==============================
# 文件路径
# ==============================
INPUT_FILE = r"C:\Users\24844\Desktop\lamost-bayesian\data\dr7_v2.0_LRS_catalogue.csv.gz"

OUTPUT_DIR = r"C:\Users\24844\Desktop\lamost-bayesian\data"

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "lamost_train.csv"
)

VAL_FILE = os.path.join(
    OUTPUT_DIR,
    "lamost_val.csv"
)


# ==============================
# 参数
# ==============================
CHUNK_SIZE = 100_000
RANDOM_STATE = 42


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("LAMOST DR7 v2.0 数据处理")
    print("=" * 60)

    print("\n开始读取数据...")

    # 分块读取，避免一次性占用大量内存
    chunks = []

    for i, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            compression="gzip",
            sep="|",
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):
        print(f"正在读取第 {i} 个数据块: {len(chunk):,} 条")
        chunks.append(chunk)

    # 合并
    print("\n正在合并数据...")
    df = pd.concat(chunks, ignore_index=True)

    print(f"总数据量: {len(df):,}")

    # ==============================
    # 7:3 划分
    # ==============================
    print("\n开始按照 7:3 划分...")

    train_df, val_df = train_test_split(
        df,
        test_size=0.3,
        random_state=RANDOM_STATE
    )

    # ==============================
    # 保存
    # ==============================
    print("\n正在保存训练集...")
    train_df.to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("训练集：")
    print(TRAIN_FILE)

    print("\n正在保存验证集...")
    val_df.to_csv(
        VAL_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("验证集：")
    print(VAL_FILE)

    # ==============================
    # 统计
    # ==============================
    total = len(train_df) + len(val_df)

    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)

    print(f"总数据: {total:,}")
    print(f"训练集: {len(train_df):,} ({len(train_df) / total:.2%})")
    print(f"验证集: {len(val_df):,} ({len(val_df) / total:.2%})")


if __name__ == "__main__":
    main()