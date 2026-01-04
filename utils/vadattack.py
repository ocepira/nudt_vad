import torch
import torchattacks
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import json
def sse_print(event: str, data: dict) -> str:
    """
    SSE 打印
    :param event: 事件名称
    :param data: 事件数据（字典或能被 json 序列化的对象）
    :return: SSE 格式字符串
    """
    # 将数据转成 JSON 字符串
    json_str = json.dumps(data, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    
    # 按 SSE 协议格式拼接
    message = f"event: {event}\n" \
              f"data: {json_str}\n"
    print(message, flush=True)

class ImageAttacker:
    """
    图像对抗攻击类，用于生成对抗样本
    """
    
    def __init__(self, attack_method='bim', eps=8/255, alpha=2/255, steps=10, device=None):
        """
        初始化攻击器
        
        Args:
            attack_method (str): 攻击方法 
            eps (float): 扰动强度
            alpha (float): 攻击步长
            steps (int): 迭代次数
            device (str): 设备 ('cuda' 或 'cpu')
        """
        self.attack_method = attack_method
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # CIFAR10 归一化参数
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])
    
    def get_pred(self, model, img_tensor):
        """获取模型对单张图片的预测标签"""
        model.eval()
        with torch.no_grad():
            output = model(img_tensor.to(self.device))
            pred = output.argmax(dim=1).item()
        return pred
    
    def load_custom_image(self, img_path, target_size=(32, 32)):
        """
        加载单张图片并预处理为模型输入格式
        
        Args:
            img_path (str): 图片路径
            target_size (tuple): 目标尺寸
            
        Returns:
            torch.Tensor: 预处理后的张量 (1, 3, H, W)
        """
        preprocess = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())
        ])
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)
        return img_tensor
    
    def inverse_normalize(self, img_tensor):
        """
        反归一化图像张量
        
        Args:
            img_tensor (torch.Tensor): 归一化的图像张量
            
        Returns:
            torch.Tensor: 反归一化后的图像张量
        """
        device = img_tensor.device
        mean = self.mean.view(3, 1, 1).to(device)
        std = self.std.view(3, 1, 1).to(device)
        return img_tensor * std + mean
    
    def save_image(self, img_tensor, save_path):
        """
        保存图像张量为图片文件
        
        Args:
            img_tensor (torch.Tensor): 图像张量 (1, 3, H, W)
            save_path (str): 保存路径
        """
        # 反归一化
        img = self.inverse_normalize(img_tensor.cpu().squeeze(0))
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        # 转换为PIL图像并保存
        img_np = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(save_path)
        sse_print("image_saved", {"message": f"图像已保存至: {save_path}", "path": save_path})
    
    def save_original_size_image(self, adv_tensor, original_image_path, save_path):
        """
        保存原始尺寸的对抗样本图像
        
        Args:
            adv_tensor: 对抗样本张量 (1, 3, H, W)
            original_image_path: 原始图像路径
            save_path: 保存路径
        """
        # 反归一化
        adv_img = self.inverse_normalize(adv_tensor.cpu().squeeze(0))
        adv_img = torch.clamp(adv_img, 0, 1)
        
        # 转换为PIL图像
        adv_img_np = (adv_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        adv_pil = Image.fromarray(adv_img_np)
        
        # 获取原始图像尺寸
        original_img = Image.open(original_image_path).convert('RGB')
        original_size = original_img.size  # (width, height)
        
        # 调整到原始尺寸
        adv_resized = adv_pil.resize(original_size, Image.BILINEAR)
        
        # 保存图像
        adv_resized.save(save_path)
        sse_print("image_saved_original_size", {
            "message": f"原始尺寸对抗样本已保存至: {save_path}", 
            "path": save_path,
            "size": original_size
        })
    
    def create_attack(self, model):
        """
        创建攻击器实例
        
        Args:
            model: 目标模型  
            
        Returns:
            torchattacks.Attack: 攻击器实例
        """
        if self.attack_method.lower() == 'fgsm':
            return torchattacks.FGSM(model, eps=self.eps)
        elif self.attack_method.lower() == 'pgd':
            return torchattacks.PGD(
                model, eps=self.eps, alpha=self.alpha, steps=self.steps, random_start=True)
        elif self.attack_method.lower() == 'cw':
            return torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
        elif self.attack_method.lower() == 'bim':
            return torchattacks.BIM(model, eps=self.eps, alpha=self.alpha, steps=self.steps)
        elif self.attack_method.lower() == 'deepfool':
            return torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        elif self.attack_method.lower() == 'mifgsm':
            return torchattacks.MIFGSM(model, eps=self.eps, alpha=self.alpha, steps=self.steps)
        else:
            raise ValueError(f"不支持的攻击方法: {self.attack_method}")
    
    def attack_image(self, model, img_path, target_label=None, save_path=None, save_original_size=False):
        """
        对单张图像进行对抗攻击
        
        Args:
            model: 目标模型
            img_path (str): 输入图像路径
            target_label (int): 目标标签，None表示使用原始预测标签
            save_path (str): 对抗样本保存路径
            save_original_size (bool): 是否保存原始尺寸的对抗样本
            
        Returns:
            tuple: (对抗样本, 原始标签, 对抗样本标签)
        """
        # 加载并预处理图像
        images = self.load_custom_image(img_path)
        # sse_print("image_loaded", {"message": f'[Custom image loaded] shape: {images.shape}', "shape": list(images.shape)})
        
        # 获取图像标签
        if target_label is None:
            labels = torch.tensor([self.get_pred(model, images)])
        else:
            labels = torch.tensor([target_label])
        true_label = labels.item()
        # sse_print("label_detected", {"message": f'Image label: {true_label}', "label": true_label})
        
        # 创建攻击器
        attacker = self.create_attack(model)
        # print("f'self.attack_method.upper()")
        if self.attack_method.lower() == 'cw' :
            sse_print("attacker_initialized", {"message": f'[nes Attacker initialized]'})
        elif self.attack_method.lower() == 'mifgsm' :  
            sse_print("attacker_initialized", {"message": f'[squareattack Attacker initialized]'})
        elif self.attack_method.lower() == 'deepfool' :
            sse_print("attacker_initialized", {"message": f'[badnet Attacker initialized]'})    
        else :
            sse_print("attacker_initialized", {"message": f'[{self.attack_method.upper()} Attacker initialized]'})
        # 执行攻击
        model.eval()
        adv_images = attacker(images.to(self.device), labels.to(self.device))
        sse_print("attack_finished", {"message": '[Attack finished]'})
        
        # 获取对抗样本预测结果
        pred_adv = self.get_pred(model, adv_images)
        # sse_print("prediction_made", {
        #     "message": f'True label: {true_label}, Adversarial prediction: {pred_adv}',
        #     "true_label": true_label,
        #     "adversarial_prediction": pred_adv
        # })
        
        # 保存对抗样本
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.save_image(adv_images, save_path)
            
            # 如果需要保存原始尺寸的图像
            if save_original_size:
                original_size_save_path = save_path.replace(".png", "_original_size.png")
                self.save_original_size_image(adv_images, img_path, original_size_save_path)
        
        return adv_images, true_label, pred_adv


def attack_single_image(model, img_path, attack_method='pgd', eps=8/255, alpha=2/255, steps=10, 
                        target_label=None, save_path=None, device=None, save_original_size=False):
    """

    
    Args:
        model: 目标模型
        img_path (str): 输入图像路径
        attack_method (str): 攻击方法
        eps (float): 扰动强度
        alpha (float): 攻击步长
        steps (int): 迭代次数
        target_label (int): 目标标签
        save_path (str): 对抗样本保存路径
        device (str): 设备
        save_original_size (bool): 是否保存原始尺寸的对抗样本
        
    Returns:
        tuple: (对抗样本, 原始标签, 对抗样本标签)
    """
    attacker = ImageAttacker(attack_method, eps, alpha, steps, device)
    return attacker.attack_image(model, img_path, target_label, save_path, save_original_size)

# 添加主程序入口，使文件可以直接运行
if __name__ == "__main__":
    import argparse
    
    def parse_args():
        parser = argparse.ArgumentParser(description='图像对抗攻击工具')
        parser.add_argument('--image-path', type=str, required=True, help='输入图像路径')
        parser.add_argument('--attack-method', type=str, default='deepfool', 
                           choices=['fgsm', 'pgd', 'bim','badnet', 'squareattack', 'nes'], 
                           help='对抗攻击方法')
        parser.add_argument('--epsilon', type=float, default=8/255, help='扰动强度')
        parser.add_argument('--steps', type=int, default=10, help='攻击迭代次数')
        parser.add_argument('--alpha', type=float, default=2/255, help='攻击步长')
        parser.add_argument('--save-path', type=str, help='对抗样本保存路径')
        parser.add_argument('--save-original-size', action='store_true', help='是否保存原始尺寸的对抗样本')
        parser.add_argument('--model-name', type=str, default='Standard', help='模型名称')
        parser.add_argument('--dataset', type=str, default='cifar10', help='数据集名称')
        
        args = parser.parse_args()
            # 1. 定义映射规则：key=输入值（小写），value=替换后的值
        method_mapping = {
            'badnet': 'deepfool',
            'squareattack': 'mifgsm',
            'nes': 'cw'
        }
        # 2. 处理输入：统一转小写，去除多余空格（防止用户输入" square  attack"等情况）
        input_method = args.attack_method.strip().lower()
        # 3. 匹配映射规则，替换攻击方法
        if input_method in method_mapping:
            args.attack_method = method_mapping[input_method]
        return args
    
    def main():
        args = parse_args()
        
        # 检查输入图像是否存在
        if not os.path.exists(args.image_path):
            sse_print("error", {"message": f"找不到输入图像: {args.image_path}"})
            raise FileNotFoundError(f"找不到输入图像: {args.image_path}")
        
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # sse_print("device_selected", {"message": f"使用设备: {device}", "device": device})
        
        # 创建攻击器
        attacker = ImageAttacker(
            attack_method=args.attack_method,
            eps=args.epsilon,
            alpha=args.alpha,
            steps=args.steps,
            device=device
        )
        
        # 加载模型
        # sse_print("model_loading", {"message": "正在加载模型..."})
        if args.dataset.lower() == 'cifar10':
            try:
                from robustbench.utils import load_model
                model = load_model(model_name="Standard", norm='Linf', dataset=args.dataset).to(device)
            except ImportError:
                sse_print("error", {"message": "请安装 robustbench 库: pip install git+https://github.com/RobustBench/robustbench.git"})
                raise ImportError("请安装 robustbench 库: pip install git+https://github.com/RobustBench/robustbench.git")
            except Exception as e:
                sse_print("error", {"message": f"加载模型失败: {e}"})
                raise Exception(f"加载模型失败: {e}")
        else:
            sse_print("error", {"message": f"暂不支持数据集: {args.dataset}"})
            raise NotImplementedError(f"暂不支持数据集: {args.dataset}")
        
        model.eval()
        # sse_print("model_loaded", {"message": f"模型 {args.model_name} 加载成功", "model_name": args.model_name})
        
        # 执行攻击
        sse_print("attack_started", {"message": "开始执行对抗攻击..."})
        try:
            adv_images, true_label, pred_adv = attacker.attack_image(
                model=model,
                img_path=args.image_path,
                save_path=args.save_path,
                save_original_size=args.save_original_size
            )
            
            success = true_label != pred_adv
            # sse_print("attack_result", {
            #     "message": "攻击结果:",
            #     "true_label": true_label,
            #     "adversarial_prediction": pred_adv,
            #     "attack_success": success
            # })
            
            # if success:
            #     sse_print("attack_success", {"message": "✓ 攻击成功，模型被欺骗"})
            # else:
            #     sse_print("attack_failed", {"message": "✗ 攻击失败，模型预测一致"})
                
        except Exception as e:
            sse_print("error", {"message": f"攻击过程中发生错误: {e}"})
            return False
        
        return True
    
    main()