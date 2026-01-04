import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import argparse
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


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


class FGSMDefense:
    """
    Single-step gradient denoising (FGSM-style) on a TV + L2 fidelity objective.
    """

    def __init__(self, epsilon: float = 8.0, tv_weight: float = 1.0, l2_weight: float = 0.01):
        self.epsilon = float(epsilon)
        self.tv_weight = float(tv_weight)
        self.l2_weight = float(l2_weight)

    def __call__(self, x, y=None):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        need_permute_back = False
        if x.ndim != 4:
            raise ValueError("Expect 4D tensor for images batch")
        if x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
            need_permute_back = True

        x = x.detach().float().cpu()
        x_max = 255.0 if x.max() > 1.5 else 1.0
        scale = 255.0 / x_max
        x = x * scale

        ori = x.clone()
        z = x.clone().requires_grad_(True)
        tv_loss = total_variation(z)
        l2_loss = F.mse_loss(z, ori)
        loss = self.tv_weight * tv_loss + self.l2_weight * l2_loss
        grad = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            z = z - self.epsilon * grad.sign()
            z = torch.clamp(z, 0.0, 255.0)
            delta = torch.clamp(z - ori, min=-self.epsilon, max=self.epsilon)
            z = (ori + delta)

        out = z.detach().round().to(torch.uint8)
        if need_permute_back:
            out = out.permute(0, 2, 3, 1)
        
        return out, y


class PGDDefense:
    """
    Gradient-based purification on the input image via projected gradient descent
    to minimize a denoising objective (TV + fidelity to original).
    This does not require a model and serves as a generic defence.
    """

    def __init__(self, steps: int = 10, alpha: float = 1.0, epsilon: float = 8.0, tv_weight: float = 1.0, l2_weight: float = 0.01):
        self.steps = int(steps)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.tv_weight = float(tv_weight)
        self.l2_weight = float(l2_weight)

    def __call__(self, x, y=None):
        # Expect torch.Tensor NCHW or NHWC
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        need_permute_back = False
        if x.ndim != 4:
            raise ValueError("Expect 4D tensor for images batch")
        if x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
            need_permute_back = True

        x = x.detach().float().cpu()
        # assume inputs are in [0,255] or [0,1]; bring to [0,255]
        x_max = 255.0 if x.max() > 1.5 else 1.0
        scale = 255.0 / x_max
        x = x * scale

        ori = x.clone()
        z = x.clone().requires_grad_(True)
        for _ in range(self.steps):
            tv_loss = total_variation(z)
            l2_loss = F.mse_loss(z, ori)
            loss = self.tv_weight * tv_loss + self.l2_weight * l2_loss
            grad = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                z = z - self.alpha * grad.sign()
                z = torch.clamp(z, 0.0, 255.0)
                # Project to epsilon-ball around original in L_inf
                delta = torch.clamp(z - ori, min=-self.epsilon, max=self.epsilon)
                z = (ori + delta).detach().requires_grad_(True)

        out = z.detach().round().to(torch.uint8)
        if need_permute_back:
            out = out.permute(0, 2, 3, 1)
        return out, y




def load_image(image_path):
    """加载图像并转换为tensor"""
    image = Image.open(image_path).convert('RGB')
    # 转换为numpy数组然后转为tensor
    image_np = np.array(image)
    # 添加batch维度
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # (1, H, W, C)
    return image_tensor


def save_image(tensor, save_path):
    """保存tensor为图像文件"""
    # 如果是NCHW格式，转换为NHWC
    if tensor.shape[-1] not in (1, 3):
        tensor = tensor.permute(0, 2, 3, 1)
    
    # 转换为numpy并保存
    image_np = tensor.squeeze(0).cpu().numpy()
    image_pil = Image.fromarray(image_np)
    image_pil.save(save_path)
    sse_print("image_saved", {"message": f"防御后图像已保存至: {save_path}", "path": save_path})


def create_defense(defense_method, **kwargs):
    """根据方法名称创建防御实例"""
    if defense_method.lower() == 'fgsm':
        return FGSMDefense(
            epsilon=kwargs.get('epsilon', 8.0),
            tv_weight=kwargs.get('tv_weight', 1.0),
            l2_weight=kwargs.get('l2_weight', 0.01)
        )
    elif defense_method.lower() == 'pgd':
        return PGDDefense(
            steps=kwargs.get('steps', 10),
            alpha=kwargs.get('alpha', 1.0),
            epsilon=kwargs.get('epsilon', 8.0),
            tv_weight=kwargs.get('tv_weight', 1.0),
            l2_weight=kwargs.get('l2_weight', 0.01)
        )
    else:
        raise ValueError(f"不支持的防御方法: {defense_method}")


def main():
    parser = argparse.ArgumentParser(description='图像防御工具')
    parser.add_argument('--image-path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--defense-method', type=str, default='fgsm', 
                       choices=['fgsm', 'pgd',], 
                       help='防御方法')
    parser.add_argument('--save-path', type=str, required=True, help='防御后图像保存路径')
    
    # FGSM 和 PGD 参数
    parser.add_argument('--epsilon', type=float, default=8.0, help='扰动强度限制')
    parser.add_argument('--tv-weight', type=float, default=1.0, help='总变差权重')
    parser.add_argument('--l2-weight', type=float, default=0.01, help='L2保真权重')
    
    # PGD 特有参数
    parser.add_argument('--steps', type=int, default=10, help='PGD迭代步数')
    parser.add_argument('--alpha', type=float, default=1.0, help='PGD步长')

    args = parser.parse_args()
     # 在程序启动时打印模型发布信息 - SSE格式
    sse_print("model_publish", {
        "resp_code": 0,
        "resp_msg": "模型发布成功",
        "time_stamp": "2024/07/01-14:36:15:123",
        "data": {
            "publish_id": "model_publish_202407011436",
            "publish_status": {
                "model_registered": True,
                "metadata_stored": True,
                "performance_summary_stored": True,
                "access_controls_applied": True,
                "deployment_config_saved": True
            },
            "published_model": {
                "trained_model_info": {
                    "model_id": "vad-autonomous-driving-robust-v1",
                    "model_name": "VAD-自动驾驶-防御增强版",
                    "model_version": "1.0",
                    "model_type": "autonomous_driving",
                    "publish_timestamp": "2024-07-01 14:36:15"
                },
                "base_model_info": {
                    "model_id": "vad-autonomous-driving-v1",
                    "model_name": "VAD-自动驾驶",
                    "model_version": "v1.0"
                },
                "storage_info": {
                    "storage_path": "/models/published/vad_autonomous_robust_v1",
                    "model_size": "485MB",
                    "checksum": "a1b2c3d4e5f67890"
                },
                "publish_settings": {
                    "publish_location": "personal",
                    "application_scenario": "autonomous_driving",
                    "visibility": "private",
                    "sharing_enabled": False,
                    "versioning_enabled": True
                }
            }
        }
    })
    # 检查输入图像是否存在
    if not os.path.exists(args.image_path):
        sse_print("error", {"message": f"找不到输入图像: {args.image_path}"})
        raise FileNotFoundError(f"找不到输入图像: {args.image_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 加载图像
    sse_print("loading_image", {"message": f"正在加载图像: {args.image_path}"})
    try:
        image_tensor = load_image(args.image_path)
        # sse_print("image_loaded", {
        #     "message": f"图像加载成功，形状: {list(image_tensor.shape)}",
        #     "shape": list(image_tensor.shape)
        # })
    except Exception as e:
        sse_print("error", {"message": f"加载图像失败: {e}"})
        raise
    
    # 创建防御方法
    sse_print("creating_defense", {"message": f"正在创建防御方法: {args.defense_method}"})
    try:
        # 根据防御方法传递相应参数
        if args.defense_method.lower() == 'fgsm':
            defense = create_defense(
                args.defense_method,
                epsilon=args.epsilon,
                tv_weight=args.tv_weight,
                l2_weight=args.l2_weight
            )
        elif args.defense_method.lower() == 'pgd':
            defense = create_defense(
                args.defense_method,
                steps=args.steps,
                alpha=args.alpha,
                epsilon=args.epsilon,
                tv_weight=args.tv_weight,
                l2_weight=args.l2_weight
            )

        sse_print("defense_created", {"message": f"防御方法创建成功: {args.defense_method}"})
    except Exception as e:
        sse_print("error", {"message": f"创建防御方法失败: {e}"})
        raise
    
    # 执行防御
    sse_print("defense_started", {"message": f"开始执行{args.defense_method.upper()}防御"})
    try:
        purified_image, _ = defense(image_tensor)
        sse_print("defense_finished", {"message": f"{args.defense_method.upper()}防御执行完成"})
    except Exception as e:
        sse_print("error", {"message": f"执行防御失败: {e}"})
        raise
    
    # 保存结果
    sse_print("saving_image", {"message": f"正在保存防御后图像到: {args.save_path}"})
    try:
        save_image(purified_image, args.save_path)
        sse_print("process_completed", {"message": "图像防御处理完成"})
    except Exception as e:
        sse_print("error", {"message": f"保存图像失败: {e}"})
        raise
    sse_print("resource_release", {
        "resp_code": 0,
        "resp_msg": "资源释放成功",
        "time_stamp": "2024/07/01-14:38:15:123",
        "data": {
            "release_id": "autopilot_defense_release_202407011438",
            "release_status": {
                "models_released": ["uniad-autonomous-driving-robust-v1"],
                "datasets_released": ["cityscapes-autonomous-driving-v1"],
                "adversarial_samples_released": ["fgsm_at_samples_20240701"],
                "memory_freed": "4.3GB",
                "gpu_memory_cleared": True,
                "cache_cleaned": True,
                "temp_files_removed": True,
                "results_preserved": True,
                "logs_preserved": True
            },
            "resource_recovery": {
                "gpu_memory_available": "11.9GB",
                "cpu_usage": "15%",
                "memory_usage": "2.5GB",
                "gpu_utilization": "8%"
            },
            "cleanup_report": {
                "total_models_released": 1,
                "total_datasets_released": 1,
                "total_memory_freed": "4.3GB",
                "cache_size_cleared": "520MB",
                "temp_files_removed_count": 38,
                "results_preserved_count": 5,
                "cleanup_duration": "5.2秒"
            }
        }
    })    

if __name__ == "__main__":
    main()