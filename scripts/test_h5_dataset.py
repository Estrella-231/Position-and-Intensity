
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from my_dataset import MyDataSetTCIR

# 加载数据集
# 如果你的 h5 文件不在当前目录，请修改路径
h5_path = "typhoon_dataset.h5"
dataset = MyDataSetTCIR(h5_path, multi_modal=True)
print(f"数据集大小：{len(dataset)}")

# 测试读取第一个样本
img, vmax, lon, lat, time, id = dataset[0]
print(f"图像形状：{np.array(img).shape}")
print(f"经度：{lon}, 纬度：{lat}")
print(f"时间：{time}")
print(f"ID：{id}")
