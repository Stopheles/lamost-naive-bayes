import os
import requests
from tqdm import tqdm


URL = (
    "https://dr7.lamost.org/catdl"
    "?name=dr7_v2.0_LRS_catalogue.csv.gz"
)

SAVE_DIR = r"C:\Users\24844\Desktop\lamost-bayesian\data"

# 下载后的压缩文件
SAVE_PATH = os.path.join(
    SAVE_DIR,
    "dr7_v2.0_LRS_catalogue.csv.gz"
)


def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        )
    }

    print("开始连接 LAMOST...")
    print("下载地址：")
    print(url)

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(30, 300)
    ) as response:

        response.raise_for_status()

        total_size = int(
            response.headers.get("Content-Length", 0)
        )

        chunk_size = 1024 * 1024  # 1 MB

        with open(save_path, "wb") as f:

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="下载进度"
            ) as pbar:

                for chunk in response.iter_content(
                    chunk_size=chunk_size
                ):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    print("\n下载完成！")
    print("文件位置：")
    print(save_path)


if __name__ == "__main__":
    download_file(
        URL,
        SAVE_PATH
    )