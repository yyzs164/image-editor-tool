# 图片处理命令行工具

## 1. 项目介绍
本项目是一个基于 Python 和 OpenCV 的命令行图片处理工具。
支持功能：读取图片基本信息、按比例缩放图片、指定坐标裁剪图片。
程序形式：命令行程序。

## 2. 环境配置
- Python 版本：3.8 及以上
- 依赖安装：运行 `pip install opencv-python`

## 3. 运行方式
启动命令：
`python main.py <图片路径> [参数]`

**功能示例：**
1. 读取信息：`python main.py demo.jpg`
2. 缩放图片：`python main.py demo.jpg --resize-width 200`
3. 裁剪图片：`python main.py demo.jpg --crop 10 10 100 100`
