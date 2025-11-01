#!/usr/bin/env python3
"""
下载ResNet34预训练权重文件
"""

import torch
import torchvision.models as models
import os

def download_resnet34_pretrained():
    """下载ResNet34预训练权重"""
    
    # 创建models目录
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # 权重文件路径
    weight_path = os.path.join(models_dir, "resnet34-333f7ec4.pth")
    
    # 检查文件是否已存在
    if os.path.exists(weight_path):
        print(f"✅ 预训练权重文件已存在: {weight_path}")
        print(f"文件大小: {os.path.getsize(weight_path) / (1024*1024):.1f} MB")
        return True
    
    print("正在下载ResNet34预训练权重...")
    print(f"保存路径: {weight_path}")
    
    try:
        # 下载预训练模型（兼容不同版本的torchvision）
        try:
            # 新版本torchvision
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except AttributeError:
            # 旧版本torchvision
            model = models.resnet34(pretrained=True)
        
        # 保存权重
        torch.save(model.state_dict(), weight_path)
        
        print("✅ 下载成功！")
        print(f"文件大小: {os.path.getsize(weight_path) / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    success = download_resnet34_pretrained()
    if success:
        print("\n🎉 预训练权重文件已准备就绪！")
        print("现在可以开始训练了。")
    else:
        print("\n💡 如果下载失败，请检查网络连接或尝试其他方法。")
