"""
读取图片的基本信息，并支持按比例缩放和裁剪。

用法示例：
    python main.py demo.jpg
    python main.py demo.jpg --width 400 -o demo_400.jpg
    python main.py demo.jpg --height 300 -o demo_300.jpg
    python main.py demo.jpg --crop 50 450 100 700 -o demo_crop.jpg
    python main.py demo.jpg --crop 50 450 100 700 --width 300 -o demo_out.jpg

裁剪坐标顺序：上、下、左、右。
下边界和右边界不包含在裁剪区域内，类似 Python 切片 [top:bottom, left:right]。
如果坐标越界，程序会自动修正到图片范围内；修正后区域无效时会给出错误提示。

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


def positive_int(text):
    """argparse 用的正整数校验函数。"""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是整数") from exc

    if value <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return value


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


def write_image(output_path, image):
    """使用 OpenCV 保存图片，并兼容包含中文的路径。"""
    ext = os.path.splitext(output_path)[1].lower()
    if not ext:
        raise ValueError("输出文件名必须包含扩展名，例如 .jpg 或 .png。")

    # JPEG 不支持透明通道，保存前先转换成 BGR
    if ext in (".jpg", ".jpeg", ".jpe"):
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim == 3 and image.shape[2] == 2:
            image = image[:, :, 0]
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        params = []

    try:
        success, encoded = cv2.imencode(ext, image, params)
    except cv2.error as exc:
        raise ValueError(f"不支持的输出图片格式：{ext}") from exc

    if not success:
        raise ValueError(f"图片编码失败，无法保存为 {ext} 格式。")

    # tofile 可以兼容中文路径
    encoded.tofile(output_path)


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


def resize_keep_aspect(image, width=None, height=None):
    """指定宽度或高度，按原图比例缩放。"""
    old_height, old_width = image.shape[:2]

    if width is not None:
        new_width = width
        new_height = max(1, round(old_height * new_width / old_width))
    else:
        new_height = height
        new_width = max(1, round(old_width * new_height / old_height))

    # 缩小图片时使用 INTER_AREA，放大时使用 INTER_LINEAR，效果更自然
    if new_width < old_width or new_height < old_height:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )


def crop_image(image, top, bottom, left, right):
    """
    按上、下、左、右坐标裁剪图片。

    下边界和右边界不包含在裁剪区域内。
    越界坐标会被修正到图片范围内。
    """
    image_height, image_width = image.shape[:2]
    requested = (top, bottom, left, right)

    # 处理越界：把坐标限制在图片范围内
    top = max(0, min(top, image_height))
    bottom = max(0, min(bottom, image_height))
    left = max(0, min(left, image_width))
    right = max(0, min(right, image_width))

    if top >= bottom:
        raise ValueError(
            "裁剪区域无效：上边界必须小于下边界，"
            f"当前修正后的坐标为 上={top}, 下={bottom}。"
        )

    if left >= right:
        raise ValueError(
            "裁剪区域无效：左边界必须小于右边界，"
            f"当前修正后的坐标为 左={left}, 右={right}。"
        )

    cropped = image[top:bottom, left:right].copy()
    corrected = (top, bottom, left, right)
    was_corrected = corrected != requested

    return cropped, corrected, was_corrected


def make_default_output_path(image_path, did_crop, did_resize):
    """没有指定输出文件时，自动生成一个不覆盖原图的文件名。"""
    stem, ext = os.path.splitext(image_path)
    if not ext:
        ext = ".jpg"

    if did_crop and did_resize:
        suffix = "_crop_resized"
    elif did_crop:
        suffix = "_crop"
    else:
        suffix = "_resized"

    return f"{stem}{suffix}{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="读取图片信息，并可按比例缩放或裁剪图片"
    )
    parser.add_argument("image", help="图片文件路径，例如 demo.jpg")

    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--width",
        type=positive_int,
        help="目标宽度（像素），高度会按原图比例自动计算",
    )
    size_group.add_argument(
        "--height",
        type=positive_int,
        help="目标高度（像素），宽度会按原图比例自动计算",
    )

    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        help=(
            "裁剪区域，依次指定 上 下 左 右 坐标；"
            "下边界和右边界不包含在区域内，越界坐标会自动修正"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="输出文件路径；省略时会自动生成，例如 demo_crop.jpg",
    )

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

    result = image
    did_crop = False
    did_resize = False

    if args.crop is not None:
        top, bottom, left, right = args.crop
        try:
            result, corrected, was_corrected = crop_image(
                result, top, bottom, left, right
            )
        except ValueError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1

        if was_corrected:
            corrected_top, corrected_bottom, corrected_left, corrected_right = corrected
            print(
                "提示：裁剪坐标已自动修正为 "
                f"上={corrected_top}, 下={corrected_bottom}, "
                f"左={corrected_left}, 右={corrected_right}。"
            )

        did_crop = True
        crop_height, crop_width = result.shape[:2]
        print(f"裁剪后尺寸：{crop_width} x {crop_height} 像素")

    if args.width is not None or args.height is not None:
        result = resize_keep_aspect(
            result,
            width=args.width,
            height=args.height,
        )
        did_resize = True
        resized_height, resized_width = result.shape[:2]
        print(f"缩放后尺寸：{resized_width} x {resized_height} 像素")

    # 没有缩放或裁剪时，只显示图片信息
    if not did_crop and not did_resize:
        return 0

    output_path = args.output
    if output_path is None:
        output_path = make_default_output_path(
            image_path, did_crop, did_resize
        )

    if os.path.abspath(output_path) == os.path.abspath(image_path):
        print(
            "错误：输出文件不能与输入文件相同，请用 -o 指定其他文件名。",
            file=sys.stderr,
        )
        return 1

    try:
        write_image(output_path, result)
    except (OSError, ValueError) as exc:
        print(f"错误：保存图片失败：{exc}", file=sys.stderr)
        return 1

    print(f"已保存：{output_path}")
    try:
        output_size = os.path.getsize(output_path)
        print(f"输出文件大小：{format_file_size(output_size)}")
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
