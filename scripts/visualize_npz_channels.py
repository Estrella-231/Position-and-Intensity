"""
将指定目录下若干 .npz 文件的每个通道保存为 PNG，便于查看通道含义。
输出目录：scripts/inspect_output/

运行：
    python scripts/visualize_npz_channels.py

会处理目录：F:\FY4_DATA\testTC_intensity 下的前 N 个 .npz（默认 N=6）
"""
from pathlib import Path
import numpy as np
import cv2
import os

NPZ_DIR = Path(r"F:\FY4_DATA\testTC_intensity")
OUT_DIR = Path(__file__).parent / 'inspect_output'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILES = 6

def normalize_to_uint8(arr):
    # arr: numpy array float32
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if amax - amin < 1e-6:
        out = np.zeros_like(arr, dtype=np.uint8)
    else:
        out = (255.0 * (arr - amin) / (amax - amin)).astype(np.uint8)
    return out


def process_file(npz_path):
    data = np.load(npz_path)
    if 'arr' not in data.files:
        print(f"跳过 (无 'arr' 键): {npz_path.name}")
        return
    arr = data['arr']  # expect (3, H, W)
    if arr.ndim != 3:
        print(f"跳过 (维度不是3): {npz_path.name} -> shape {arr.shape}")
        return
    C, H, W = arr.shape
    base = npz_path.stem
    # 保存每个通道
    for i in range(C):
        ch = arr[i]
        ch8 = normalize_to_uint8(ch)
        out_path = OUT_DIR / f"{base}_ch{i}.png"
        cv2.imwrite(str(out_path), ch8)
        print(f"保存: {out_path}")
    # 尝试合成 RGB：如果 C>=3 就用前三个通道为 R,G,B
    if C >= 3:
        r = normalize_to_uint8(arr[0])
        g = normalize_to_uint8(arr[1])
        b = normalize_to_uint8(arr[2])
        rgb = np.stack([r, g, b], axis=-1)  # H,W,3
        out_rgb = OUT_DIR / f"{base}_RGB.png"
        # cv2.imwrite expects BGR, so convert RGB -> BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_rgb), bgr)
        print(f"保存合成RGB: {out_rgb}")
    else:
        print(f"文件 {npz_path.name} 通道数 {C} < 3，未合成 RGB")


def main():
    files = sorted(list(NPZ_DIR.glob('*.npz')))
    if not files:
        print(f"目录没有找到 .npz 文件: {NPZ_DIR}")
        return
    files = files[:MAX_FILES]
    print(f"将处理 {len(files)} 个文件，输出到 {OUT_DIR}")
    for p in files:
        try:
            process_file(p)
        except Exception as e:
            print(f"处理 {p.name} 时出错: {e}")

if __name__ == '__main__':
    main()
