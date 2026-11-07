"""
读取图片的基本信息，并打印到控制台。

用法：
    python main.py demo.jpg

运行前请先安装 OpenCV：
    pip install opencv-python
"""

import argparse
import os
import sys

import cv2
import numpy as np


# 常见图片扩展名对应的格式名称
FORMAT_NAMES = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".jpe": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".dib": "BMP",
    ".gif": "GIF",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
    ".jp2": "JPEG 2000",
    ".j2k": "JPEG 2000",
    ".pbm": "PBM",
    ".pgm": "PGM",
    ".ppm": "PPM",
    ".pnm": "PNM",
    ".exr": "EXR",
    ".hdr": "HDR",
    ".pic": "PIC",
}


def get_image_format(image_path):
    """根据文件扩展名获取图片格式。"""
    ext = os.path.splitext(image_path)[1].lower()
    return FORMAT_NAMES.get(ext, ext.lstrip(".").upper() or "未知")


def read_image(image_path):
    """使用 OpenCV 读取图片，并兼容包含中文的路径。"""
    # IMREAD_UNCHANGED 会保留透明通道，便于识别 RGBA 图片
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is not None:
        return image

    # 在某些系统上 cv2.imread 无法读取中文路径，改用 imdecode 再试一次
    try:
        file_data = np.fromfile(image_path, dtype=np.uint8)
        return cv2.imdecode(file_data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def get_color_mode(image):
    """根据 OpenCV 图像的通道数判断颜色模式。"""
    channels = 1 if image.ndim == 2 else image.shape[2]

    if channels == 1:
        return "灰度（Grayscale）"
    if channels == 2:
        return "灰度 + Alpha（GA）"
    if channels == 3:
        return "RGB（OpenCV 内部按 BGR 顺序存储）"
    if channels == 4:
        return "RGBA（OpenCV 内部按 BGRA 顺序存储）"
    return f"未知（{channels} 通道）"


def format_file_size(size_bytes):
    """把字节数格式化为 KB 或 MB，并保留 2 位小数。"""
    size_kb = size_bytes / 1024

    if size_kb >= 1024:
        return f"{size_kb / 1024:.2f} MB"
    return f"{size_kb:.2f} KB"


def main():
    parser = argparse.ArgumentParser(description="读取并显示图片的基本信息")
    parser.add_argument("image", help="图片文件路径，例如 demo.jpg")
    args = parser.parse_args()

    image_path = args.image

    if not os.path.isfile(image_path):
        print(f"错误：找不到文件：{image_path}", file=sys.stderr)
        return 1

    try:
        file_size = os.path.getsize(image_path)
    except OSError as exc:
        print(f"错误：无法获取文件大小：{exc}", file=sys.stderr)
        return 1

    image = read_image(image_path)
    if image is None:
        print(
            "错误：无法读取图片，请确认文件存在且格式受 OpenCV 支持。",
            file=sys.stderr,
        )
        return 1

    height, width = image.shape[:2]

    print(f"文件名：{os.path.basename(image_path)}")
    print(f"图片格式：{get_image_format(image_path)}")
    print(f"宽度和高度：{width} x {height} 像素")
    print(f"颜色模式：{get_color_mode(image)}")
    print(f"文件大小：{format_file_size(file_size)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
