"""
查看 .npz 文件的内容结构
"""
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(message)s')

def inspect_npz(npz_path):
    """检查 .npz 文件的内容结构"""
    logging.info(f"\n{'='*50}")
    logging.info(f"检查文件: {npz_path}")
    logging.info(f"{'='*50}\n")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # 1. 显示所有键名
        logging.info("1. 文件中包含的键:")
        logging.info(f"   {data.files}\n")
        
        # 2. 检查每个键的内容
        logging.info("2. 每个键的详细信息:")
        for key in data.files:
            arr = data[key]
            logging.info(f"\n键名: {key}")
            logging.info(f"类型: {type(arr)}")
            
            # 显示形状（如果是数组）
            if hasattr(arr, 'shape'):
                logging.info(f"形状: {arr.shape}")
                logging.info(f"数据类型: {arr.dtype}")
                
                # 如果是结构化数组，显示字段名
                if arr.dtype.names is not None:
                    logging.info(f"结构化数组字段: {arr.dtype.names}")
            
            # 如果是对象数组，尝试显示第一个元素的信息
            if arr.dtype == np.dtype('O'):
                sample = arr[0] if len(arr) > 0 else None
                logging.info(f"第一个元素类型: {type(sample)}")
                if isinstance(sample, dict):
                    logging.info(f"字典键: {list(sample.keys())}")
                elif isinstance(sample, (list, tuple)):
                    logging.info(f"序列长度: {len(sample)}")
            
            # 显示数值范围（如果适用）
            try:
                if np.issubdtype(arr.dtype, np.number):
                    logging.info(f"数值范围: [{arr.min()}, {arr.max()}]")
            except:
                pass
            
            logging.info("-" * 40)
        
    except Exception as e:
        logging.error(f"检查文件时出错: {e}")
    finally:
        if 'data' in locals():
            data.close()

def main():
    # 指定 .npz 文件所在目录
    npz_dir = r"F:\FY4_DATA\testTC_intensity"
    
    # 获取目录下所有 .npz 文件
    npz_files = list(Path(npz_dir).glob("*.npz"))
    
    if not npz_files:
        logging.error(f"在 {npz_dir} 中没有找到 .npz 文件！")
        return
    
    # 显示找到的文件数量
    logging.info(f"找到 {len(npz_files)} 个 .npz 文件")
    
    # 检查每个文件
    for npz_file in npz_files:
        inspect_npz(npz_file)

if __name__ == "__main__":
    main()