import sys
sys.path.insert(0, '..')
import torch
import torch.nn as nn
import torchattacks
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# ===================== 1. 补全缺失的辅助函数 =====================


def get_pred(model, img_tensor, device):
    """获取模型对单张图片的预测标签"""
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
        pred = output.argmax(dim=1).item()
    return pred

def imshow(img_tensor, title="", save_path=None):
    """可视化张量格式的图片（反归一化）"""
    # CIFAR10 归一化参数（与模型训练一致）
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(3,1,1)
    
    # 反归一化 + 转维度 (C,H,W) → (H,W,C)
    img = img_tensor.cpu().squeeze(0) * std + mean
    img = torch.clamp(img, 0, 1)  # 限制像素值范围
    img = img.permute(1,2,0).numpy()
    
    # 绘图
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    
    # 保存图像（如果指定了保存路径）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"图像已保存至: {save_path}")
    
    plt.show()

def save_original_size_image(adv_tensor, original_image_path, save_path, target_size=(32,32)):
    """
    保存原始尺寸的对抗样本图像
    :param adv_tensor: 对抗样本张量 (1, 3, 32, 32)
    :param original_image_path: 原始图像路径
    :param save_path: 保存路径
    :param target_size: 模型输入尺寸
    """
    # CIFAR10 归一化参数
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(3,1,1)
    
    # 反归一化
    adv_img = adv_tensor.cpu().squeeze(0) * std + mean
    adv_img = torch.clamp(adv_img, 0, 1)  # 限制像素值范围
    
    # 转换为PIL图像 (C,H,W) → (H,W,C) → (0-255 uint8)
    adv_img_np = (adv_img.permute(1,2,0).numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_img_np)
    
    # 获取原始图像尺寸
    original_img = Image.open(original_image_path).convert('RGB')
    original_size = original_img.size  # (width, height)
    
    # 调整到原始尺寸
    adv_resized = adv_pil.resize(original_size, Image.BILINEAR)
    
    # 保存图像
    adv_resized.save(save_path)
    print(f"原始尺寸对抗样本已保存至: {save_path}")

# ===================== 2. 加载并预处理单张自定义图片 =====================
def load_custom_image(img_path, target_size=(32,32)):
    """
    加载单张图片并预处理为 CIFAR10 模型的输入格式
    :param img_path: 图片路径（如 './test_img.png'）
    :param target_size: 模型输入尺寸（CIFAR10 为 32x32）
    :return: 预处理后的张量 (1, 3, 32, 32)
    """
    # 预处理管道（与 CIFAR10 模型训练时一致）
    preprocess = transforms.Compose([
        transforms.Resize(target_size),  # 调整尺寸
        transforms.ToTensor(),           # 转张量 (0-1)
        transforms.Normalize(            # 归一化（CIFAR10 均值/std）
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # 加载图片（转 RGB，避免灰度图问题）
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # 添加 batch 维度 (1,3,32,32)
    return img_tensor

# ===================== 3. 核心攻击逻辑（适配单张图片） =====================
if __name__ == "__main__":
    # ---------- 配置参数 ----------
    img_path = "/data6/user24215461/autodrive/VAD/data/nuscenes/sweeps/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603620482.jpg"  # 替换为你的图片路径（如 jpg/png 格式）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_label = None  # 若为 None，自动用模型预测的标签作为攻击目标
    # PGD 攻击参数（保持与原代码一致）
    eps = 8/255
    alpha = 2/255  # 原代码笔误：2/25 → 修正为 2/255
    steps = 10
    random_start = True

    # 创建保存对抗样本的目录
    save_dir = "adversarial_examples"
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 加载模型（CIFAR10 Standard 模型） ----------
    from robustbench.utils import load_model
    model = load_model('Standard', norm='Linf').to(device)
    model.eval()  # 攻击时模型需设为 eval 模式
    print('[Model loaded]')

    # ---------- 加载并预处理自定义图片 ----------
    images = load_custom_image(img_path)  # (1,3,32,32)
    print('[Custom image loaded] shape:', images.shape)

    # ---------- 获取图片标签（手动指定/模型预测） ----------
    if target_label is None:
        labels = torch.tensor([get_pred(model, images, device)])  # 自动预测标签
    else:
        labels = torch.tensor([target_label])  # 手动指定标签
    print(f'Image label: {labels.item()}')

    # ---------- 验证原始准确率 ----------
    # acc = clean_accuracy(model, images.to(device), labels.to(device))
    # print('Clean Acc: %2.2f %%' % (acc*100))

    # ---------- 初始化 PGD 攻击 ----------
    atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=random_start)
    print('[PGD Attacker initialized]', atk)

    # ---------- 执行攻击（单张图片） ----------
    adv_images = atk(images, labels)
    print('[Attack finished]')

    # ---------- 可视化攻击结果 ----------
    idx = 0
    pred_adv = get_pred(model, adv_images[idx:idx+1], device)
    true_label = labels[idx].item()
    
    # 生成保存路径
    original_filename = os.path.basename(img_path)
    name_without_ext = os.path.splitext(original_filename)[0]
    save_path = os.path.join(save_dir, f"{name_without_ext}_adversarial.png")
    
    # 显示并保存32x32尺寸的对抗样本图像
    imshow(adv_images[idx:idx+1], title=f"True:{true_label}, Adv Pre:{pred_adv}", save_path=save_path)
    
    # 保存原始尺寸的对抗样本图像
    original_size_save_path = os.path.join(save_dir, f"{name_without_ext}_adversarial_original_size.png")
    save_original_size_image(adv_images[idx:idx+1], img_path, original_size_save_path)

    # ---------- 验证攻击后准确率 ----------
    # adv_acc = clean_accuracy(model, adv_images.to(device), labels.to(device))
    # print('Adversarial Acc: %2.2f %%' % (adv_acc*100))