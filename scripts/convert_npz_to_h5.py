"""
将 .npz 格式的台风数据转换为项目可用的 HDF5 格式。
处理 .npz 文件名格式: YYYYMMDDHHMM_lat{N/S}_lon{E/W}.npz
"""

import os
import re
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import logging
from datetime import datetime
import cv2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_filename(filename):
    """
    从文件名解析时间和位置信息
    例如：202312181200_7.7N_125.1E.npz
    """
    basename = Path(filename).stem  # 移除.npz后缀
    
    # 解析时间、纬度和经度
    pattern = r'(\d{12})_(-?\d+\.?\d*)(N|S)_(-?\d+\.?\d*)(E|W)'
    match = re.match(pattern, basename)
    
    if not match:
        raise ValueError(f"文件名格式不正确: {basename}")
    
    datetime_str, lat, ns, lon, ew = match.groups()
    
    # 转换时间
    time = datetime.strptime(datetime_str, '%Y%m%d%H%M')
    
    # 转换经纬度（考虑南纬和西经为负值）
    lat = float(lat) * (-1 if ns == 'S' else 1)
    lon = float(lon) * (-1 if ew == 'W' else 1)
    
    return {
        'time': time.strftime('%Y-%m-%d %H:%M'),  # 格式化时间字符串
        'lat': lat,
        'lon': lon,
        'ID': basename  # 使用完整文件名作为ID
    }

def load_and_process_npz(npz_path):
    """
    加载并处理单个.npz文件
    """
    # 解析文件名中的信息
    info = parse_filename(npz_path)
    
    # 加载数组数据
    with np.load(npz_path) as data:
        # 加载图像数据 (3, 256, 256)
        matrix = data['arr']
        
        # 调整图像尺寸到 201x201
        resized = np.zeros((3, 201, 201), dtype=np.float32)
        for i in range(3):
            resized[i] = cv2.resize(matrix[i], (201, 201))
        
        # 转置为 (201, 201, 3) 以匹配 MyDataSetTCIR 的期望格式
        matrix = np.transpose(resized, (1, 2, 0))
        
        # 归一化到0-255范围并转换为uint8
        matrix = np.clip(matrix * 255, 0, 255).astype(np.uint8)
        
        return matrix, info

def main():
    # 输入输出路径配置
    npz_dir = r"F:\FY4_DATA\testTC_intensity"  # .npz文件目录
    out_h5 = "typhoon_dataset.h5"              # 输出的HDF5文件路径
    
    # 获取所有.npz文件
    npz_files = list(Path(npz_dir).glob("*.npz"))
    if not npz_files:
        logging.error(f"在 {npz_dir} 中没有找到.npz文件！")
        return
    
    logging.info(f"找到 {len(npz_files)} 个.npz文件")
    
    # 处理所有文件
    matrices = []
    infos = []
    
    for npz_file in npz_files:
        try:
            matrix, info = load_and_process_npz(npz_file)
            matrices.append(matrix)
            infos.append(info)
            logging.info(f"处理完成: {npz_file.name}")
        except Exception as e:
            logging.error(f"处理文件 {npz_file.name} 时出错: {e}")
            continue
    
    if not matrices:
        logging.error("没有成功处理任何文件！")
        return
        
    # 转换为数组和DataFrame
    matrices = np.stack(matrices, axis=0)  # (N, 201, 201, 3)
    info_df = pd.DataFrame(infos)
    
    # 添加缺失的Vmax列（如果数据中没有，暂时填充0）
    if 'Vmax' not in info_df.columns:
        info_df['Vmax'] = 0  # TODO: 如果有强度数据，在这里添加
    
    # 保存为HDF5
    logging.info(f"保存到: {out_h5}")
    
    try:
        # 保存info
        info_df.to_hdf(out_h5, key='info', mode='w')
        
        # 保存matrix
        with h5py.File(out_h5, 'a') as f:
            if 'matrix' in f:
                del f['matrix']
            f.create_dataset('matrix', data=matrices, compression='gzip')
        
        # 验证生成的文件
        logging.info("验证生成的文件...")
        # 测试info读取
        info = pd.read_hdf(out_h5, 'info')
        logging.info(f"成功读取info DataFrame，形状: {info.shape}")
        # 测试matrix读取
        with h5py.File(out_h5, 'r') as f:
            matrix = f['matrix']
            logging.info(f"成功读取matrix dataset，形状: {matrix.shape}")
        
        logging.info("转换完成！")
        logging.info(f"样本数量: {len(info_df)}")
        logging.info(f"数据形状: {matrices.shape}")
        
    except Exception as e:
        logging.error(f"保存文件时出错: {e}")
        if os.path.exists(out_h5):
            os.remove(out_h5)
            logging.info(f"已删除未完成的输出文件: {out_h5}")

if __name__ == "__main__":
    main()