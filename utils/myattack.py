import sys
import os
import copy
import argparse
import warnings
import logging
from contextlib import contextmanager
from io import StringIO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.VAD.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
from mmcv.parallel import DataContainer as DC

# 全局配置：忽略冗余警告
warnings.filterwarnings("ignore")
logging.getLogger('mmcv').setLevel(logging.WARNING)
logging.getLogger('mmcv.runner').setLevel(logging.WARNING)


@contextmanager
def suppress_stdout():
    """抑制标准输出（加载权重时使用）"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


class VADAttacker:
    """适配VAD模型的对抗攻击类（重构原attacks类）"""
    def __init__(self, args, cfg, model, dataset, dataloader):
        """
        初始化攻击器（复用main函数的VAD实例，避免重复加载）
        :param args: 命令行参数
        :param cfg: VAD配置文件对象
        :param model: 已加载权重的VAD模型
        :param dataset: VAD测试数据集
        :param dataloader: VAD测试数据加载器
        """
        self.args = args
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        
        # 自动获取模型设备（GPU/CPU）
        self.device = torch.device('cpu')
        if next(model.parameters()).is_cuda:
            self.device = next(model.parameters()).device
        
        # 模型配置：train模式保留梯度，但冻结BN层（避免影响攻击）
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                m.eval()
        
        # VAD图像归一化参数（从配置中读取，避免硬编码）
        try:
            self.img_mean = torch.tensor(cfg.img_norm_cfg['mean']).reshape(1, 1, 3, 1, 1).to(self.device)
            self.img_std = torch.tensor(cfg.img_norm_cfg['std']).reshape(1, 1, 3, 1, 1).to(self.device)
        except:
            # 如果无法获取归一化参数，使用默认值
            self.img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1).to(self.device)
            self.img_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1).to(self.device)

    # def get_multiview_images(self, batch):
    #     """提取VAD多视角图像 [B, N, C, H, W]"""
    #     print(f"get_multiview_images输入类型: {type(batch)}")
        
    #     # 处理列表形式的batch
    #     if isinstance(batch, (list, tuple)):
    #         print(f"处理列表batch，长度: {len(batch)}")
    #         if len(batch) > 0:
    #             batch = batch[0]
    #             print(f"取第一个元素，类型: {type(batch)}")
    #         else:
    #             print("错误：空列表batch")
    #             return None
        
    #     # 处理字典形式的batch
    #     if isinstance(batch, dict):
    #         print(f"处理字典batch，键: {list(batch.keys())}")
    #         if 'img' not in batch:
    #             print("错误：batch中没有'img'键")
    #             return None
                
    #         img_data = batch['img']
    #         print(f"img_data类型: {type(img_data)}")
            
    #         # VAD项目中，img_data直接是列表，包含多视角图像张量
    #         if isinstance(img_data, list):
    #             print(f"img_data是列表，长度: {len(img_data)}")
    #             # 检查列表中的元素是否为张量
    #             if len(img_data) > 0 and isinstance(img_data[0], torch.Tensor):
    #                 try:
    #                     # 将图像列表stack成tensor [N, C, H, W]
    #                     stacked = torch.stack(img_data, dim=0)
    #                     print(f"stack成功，形状: {stacked.shape}")
    #                     # 添加batch维度 [1, N, C, H, W]
    #                     if len(stacked.shape) == 4:
    #                         stacked = stacked.unsqueeze(0)
    #                     return stacked.to(self.device)
    #                 except Exception as e:
    #                     print(f"Stack图像失败: {e}")
    #                     return None
    #             else:
    #                 print(f"列表中的元素不是张量: {type(img_data[0]) if len(img_data) > 0 else '空列表'}")
    #                 return None
            
    #         # 处理DataContainer或者其他封装类型（备用）
    #         elif hasattr(img_data, 'data'):
    #             print(f"img_data有data属性，类型: {type(img_data.data)}")
    #             img_content = img_data.data
    #             if isinstance(img_content, list) and len(img_content) > 0:
    #                 print(f"img_content是列表，长度: {len(img_content)}")
    #                 actual_img_data = img_content[0]
    #                 print(f"actual_img_data类型: {type(actual_img_data)}")
    #                 if isinstance(actual_img_data, torch.Tensor):
    #                     print(f"成功提取张量，形状: {actual_img_data.shape}")
    #                     return actual_img_data.to(self.device)
    #                 elif isinstance(actual_img_data, (list, tuple)):
    #                     print(f"actual_img_data是列表/元组，长度: {len(actual_img_data)}")
    #                     if all(isinstance(x, torch.Tensor) for x in actual_img_data):
    #                         try:
    #                             stacked = torch.stack(actual_img_data, dim=0)
    #                             print(f"stack成功，形状: {stacked.shape}")
    #                             return stacked.to(self.device)
    #                         except Exception as e:
    #                             print(f"Stack图像失败: {e}")
    #                             return None
    #             else:
    #                 print(f"img_content不是列表或为空: {img_content}")
    #         else:
    #             # 如果不是DataContainer，直接处理
    #             print("img_data没有data属性，直接处理")
    #             if isinstance(img_data, torch.Tensor):
    #                 print(f"img_data是张量，形状: {img_data.shape}")
    #                 return img_data.to(self.device)
    #             else:
    #                 print(f"img_data不是张量: {type(img_data)}")
    #                 return None
                    
    #     # 处理直接是tensor的情况
    #     elif isinstance(batch, torch.Tensor):
    #         print(f"batch直接是张量，形状: {batch.shape}")
    #         return batch.to(self.device)
    #     else:
    #         print(f"未知的batch类型: {type(batch)}")
            
    #     print("get_multiview_images返回None")
    #     return None
    def get_multiview_images(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            batch = batch[0]
        
        if isinstance(batch, dict) and 'img' in batch:
            img_data = batch['img']
        
        # 核心修改：处理DataContainer类型（VAD标准格式）
            if isinstance(img_data, DC):
                # img_data.data是[tensor1, tensor2, ...]，每个tensor是[N, C, H, W]
                img_content = img_data.data
                if isinstance(img_content, list) and len(img_content) > 0:
                    try:
                        # 堆叠成[B, N, C, H, W]
                        stacked = torch.stack(img_content, dim=0)
                        return stacked.to(self.device)
                    except Exception as e:
                        print(f"图像堆叠失败: {e}")
                        return None
        
        # 备用：直接张量
            elif isinstance(img_data, torch.Tensor):
                return img_data.to(self.device)
        return None
    
    # def save_vad_adv_image(self, adv_img, ori_im_files):
    #     """保存对抗样本图像（反归一化后）"""
    #     os.makedirs('vad_adv_samples', exist_ok=True)
    #     # 仅保存前视摄像头（index=0）
    #     if len(adv_img.shape) == 5:  # [B, N, C, H, W]
    #         front_view_adv = adv_img[:, 0, :, :, :]  # [B, C, H, W]
    #     else:  # [B, C, H, W]
    #         front_view_adv = adv_img
            
    #     # 反归一化：adv_img = adv_img * std + mean
    #     front_view_adv = front_view_adv * self.img_std.squeeze(1) + self.img_mean.squeeze(1)
    #     front_view_adv = torch.clamp(front_view_adv, 0, 255).cpu().numpy()
        
    #     # 遍历批次保存
    #     for idx, im_file in enumerate(ori_im_files):
    #         save_name = f"adv_{os.path.basename(im_file)}"
    #         save_path = os.path.join('vad_adv_samples', save_name)
    #         # 转换维度：[C, H, W] → [H, W, C]
    #         mmcv.imwrite(front_view_adv[idx].transpose(1, 2, 0).astype(np.uint8), save_path)
    #     print(f"对抗样本已保存至: vad_adv_samples/{save_name}")

    def get_multiview_images(self, batch):
        """提取VAD多视角图像 [B, N, C, H, W]（适配DataContainer）"""
        print(f"get_multiview_images输入类型: {type(batch)}")
        
        # 处理列表形式的batch
        if isinstance(batch, (list, tuple)):
            print(f"处理列表batch，长度: {len(batch)}")
            if len(batch) > 0:
                batch = batch[0]
            else:
                print("错误：空列表batch")
                return None
        
        # 处理字典形式的batch
        if isinstance(batch, dict):
            print(f"处理字典batch，键: {list(batch.keys())}")
            if 'img' not in batch:
                print("错误：batch中没有'img'键")
                return None
                
            img_data = batch['img']
            print(f"img_data原始类型: {type(img_data)}")
            
            # 核心：解析DataContainer
            if isinstance(img_data, DC):
                print("解析DataContainer类型的img数据")
                img_content = img_data.data  # 提取真实数据
                img_data = img_content if isinstance(img_content, list) else [img_content]
            
            # 处理列表形式的img数据
            if isinstance(img_data, list):
                print(f"img_data列表长度: {len(img_data)}")
                # 过滤并提取张量
                tensor_list = []
                for item in img_data:
                    if isinstance(item, DC):
                        tensor_list.append(item.data)
                    elif isinstance(item, torch.Tensor):
                        tensor_list.append(item)
                
                if len(tensor_list) > 0 and all(isinstance(x, torch.Tensor) for x in tensor_list):
                    try:
                        stacked = torch.stack(tensor_list, dim=0)  # [N, C, H, W]
                        print(f"Stack图像成功，形状: {stacked.shape}")
                        return stacked.unsqueeze(0).to(self.device)  # 加batch维度 [1, N, C, H, W]
                    except Exception as e:
                        print(f"Stack失败: {e}")
                        return None
                else:
                    print("列表中无有效张量")
                    return None
            
            # 直接处理张量
            elif isinstance(img_data, torch.Tensor):
                print(f"img_data是张量，形状: {img_data.shape}")
                return img_data.unsqueeze(0).to(self.device) if len(img_data.shape)==4 else img_data.to(self.device)
                
        elif isinstance(batch, torch.Tensor):
            print(f"batch是张量，形状: {batch.shape}")
            return batch.to(self.device)
        
        print("无法提取图像，返回None")
        return None    

    def pgd_attack(self, batch):
        """
        PGD对抗攻击（适配VAD多视角图像+核心损失）
        :param batch: VAD完整批次数据
        :return: 带对抗扰动的batch
        """
        # 1. 提取多视角图像并备份原始数据
        img = self.get_multiview_images(batch)
        
        if img is None:
            print("警告：无法提取图像数据，返回原始batch")
            return batch
        
        print(f"成功提取图像，形状: {img.shape}, 类型: {img.dtype}, 设备: {img.device}")
        
        # 2. 设置攻击参数
        eps = self.args.epsilon
        alpha = self.args.step_size

        # 3. 随机初始化扰动
        img_ori = img.clone().detach()
        adv_img = img.clone().detach()
        if self.args.random_start:
            delta = torch.empty_like(adv_img).uniform_(-eps, eps)
            adv_img = adv_img + delta
            adv_img = torch.clamp(adv_img, 0, 1)

        # 4. PGD迭代攻击
        for i in range(self.args.max_iterations):
            adv_img.requires_grad = True
            
            # 构造用于攻击的batch副本
            attack_batch = None
            if isinstance(batch, list):
                attack_batch = []
                for b in batch:
                    if isinstance(b, dict):
                        attack_batch.append(b.copy())
                    else:
                        attack_batch.append(b)
                if len(attack_batch) > 0 and isinstance(attack_batch[0], dict):
                    attack_batch[0]['img'] = adv_img
            elif isinstance(batch, dict):
                attack_batch = batch.copy()
                attack_batch['img'] = adv_img
            else:
                attack_batch = adv_img

            # 4.1 VAD模型前向传播
            try:
                if isinstance(attack_batch, dict):
                    outputs = self.model(return_loss=True, **attack_batch)
                elif isinstance(attack_batch, list):
                    outputs = self.model(attack_batch, return_loss=True)
                else:
                    outputs = self.model(attack_batch, return_loss=True)
            except Exception as e:
                print(f"模型前向传播失败: {e}")
                return batch

            # 4.2 计算损失
            loss = None
            if isinstance(outputs, dict):
                print(f"模型输出字典键: {list(outputs.keys())}")
                # 尝试多种可能的损失键
                for loss_key in ['loss', 'loss_total', 'total_loss']:
                    if loss_key in outputs:
                        loss = outputs[loss_key]
                        print(f"使用损失键: {loss_key}, 损失值: {loss}")
                        break
                        
                # 如果找不到直接的损失键，尝试组合多个损失项
                if loss is None:
                    loss_components = []
                    for key, value in outputs.items():
                        if 'loss' in key.lower() and isinstance(value, torch.Tensor):
                            loss_components.append(value)
                    if loss_components:
                        loss = sum(loss_components)
                        print(f"组合损失: {loss}")
                        
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
                print(f"直接损失张量: {loss}")
                
            if loss is None:
                print("无法获取模型损失")
                return batch

            # 4.3 计算梯度并更新对抗样本
            self.model.zero_grad()
            try:
                loss.backward()
            except Exception as e:
                print(f"反向传播失败: {e}")
                return batch
            
            if adv_img.grad is None:
                print("无法计算梯度")
                return batch
                
            grad = adv_img.grad.data

            # 梯度上升（最大化损失，攻击模型）
            adv_img = adv_img.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_img - img_ori, min=-eps, max=eps)
            adv_img = img_ori + delta
            adv_img = torch.clamp(adv_img, 0, 1).detach()

        # 5. 替换batch中的图像为对抗样本
        # try:
        #     if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
        #         batch[0]['img'] = adv_img
        #     elif isinstance(batch, dict):
        #         batch['img'] = adv_img
        #     print("成功替换batch中的图像为对抗样本")
        # except Exception as e:
        #     print(f"替换图像时出错: {e}")
        # try:
        #     if isinstance(batch, list) and len(batch) > 0:
        #         for i, b in enumerate(batch):
        #             if isinstance(b, dict) and 'img' in b:
        #         # 关键修改：用DC包装对抗样本，保持原始数据结构
        #                 batch[i]['img'] = DC([adv_img[i:i+1]], stack=True, pad_dims=3)
        #     elif isinstance(batch, dict):
        # # 关键修改：用DC包装对抗样本，保持原始数据结构
        #         batch['img'] = DC([adv_img], stack=True, pad_dims=3)
        #     print("成功替换batch中的图像为对抗样本")
        # except Exception as e:
        #     print(f"替换图像时出错: {e}")
        # return batch
            # 5. 替换batch中的图像为对抗样本（适配DataContainer）
        try:
            adv_img_squeezed = adv_img.squeeze(0)  # 去掉batch维度 [N, C, H, W]
            if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
                # 恢复为DataContainer格式
                batch[0]['img'] = DC(adv_img_squeezed, stack=True, padding_value=0)
            elif isinstance(batch, dict):
                batch['img'] = DC(adv_img_squeezed, stack=True, padding_value=0)
            print("成功替换batch中的图像为对抗样本（保留DataContainer格式）")
        except Exception as e:
            print(f"替换图像出错: {e}")
            import traceback
            traceback.print_exc()

    # def run_attack(self):
    #     """执行攻击，生成带对抗样本的数据集"""
    #     attacked_batches = []
    #     print(f"\n开始生成对抗样本（共{self.args.gen_adv_sample_num}个）...")
        
    #     success_count = 0
    #     for batch_idx, batch in enumerate(self.dataloader):
    #         if success_count >= self.args.gen_adv_sample_num:
    #             break
                
    #         print(f"\n处理第{batch_idx+1}个批次...")
    #         print(f"批次类型: {type(batch)}")
    #         if hasattr(batch, 'keys'):
    #             print(f"批次键: {list(batch.keys())}")
            
    #         try:
    #             # 执行PGD攻击
    #             if self.args.attack_method == 'pgd':
    #                 attacked_batch = self.pgd_attack(batch)
    #                 if attacked_batch is not None:
    #                     attacked_batches.append(attacked_batch)
    #                     success_count += 1
    #                     print(f"成功生成第{success_count}个对抗样本")
    #                 else:
    #                     print("警告：pgd_attack返回None，使用原始批次")
    #                     attacked_batches.append(batch)
    #                     success_count += 1
    #             else:
    #                 raise NotImplementedError(f"暂不支持攻击方法：{self.args.attack_method}")
    #         except Exception as e:
    #             print(f"处理批次时出错: {e}")
    #             import traceback
    #             traceback.print_exc()
    #             # 即使出错也继续处理下一个批次
    #             attacked_batches.append(batch)  # 添加原始批次以保证数量
    #             success_count += 1
    #             continue

    #     print(f"\n对抗样本生成完成！共生成{len(attacked_batches)}个批次")
    #     return attacked_batches

    def run_attack(self):
        attacked_batches = []
        print(f"\n开始生成对抗样本（共{self.args.gen_adv_sample_num}个）...")
        
        success_count = 0
        for batch_idx, batch in enumerate(self.dataloader):
            if success_count >= self.args.gen_adv_sample_num:
                break
                
            print(f"\n处理第{batch_idx+1}个批次...")
            print(f"批次类型: {type(batch)}")
            
            try:
                # 执行PGD攻击
                if self.args.attack_method == 'pgd':
                    attacked_batch = self.pgd_attack(batch)
                    attacked_batches.append(attacked_batch if attacked_batch else batch)
                    success_count += 1
                    print(f"成功生成第{success_count}个对抗样本")
                else:
                    raise NotImplementedError(f"暂不支持{self.args.attack_method}")
            except Exception as e:
                print(f"批次处理出错: {e}")
                import traceback
                traceback.print_exc()
                attacked_batches.append(batch)
                success_count += 1

        print(f"\n对抗样本生成完成！共生成{len(attacked_batches)}个批次")
        
        # 核心：包装为合法DataLoader（保持原生接口）
        class AdvDataLoader:
            def __init__(self, batches):
                self.batches = batches
                self.batch_size = 1
                self.num_workers = 0
                self.pin_memory = False
                
            def __iter__(self):
                return iter(self.batches)
            
            def __len__(self):
                return len(self.batches)
        
        return AdvDataLoader(attacked_batches)


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='VAD模型对抗攻击测试')
    # 基础参数
    parser.add_argument('config', help='VAD配置文件路径')
    parser.add_argument('checkpoint', help='VAD权重文件路径')
    parser.add_argument('--launcher', default='none', help='分布式启动器 (none/distributed)')
    parser.add_argument('--eval', default='bbox', help='评估指标 (bbox/map/traj)')
    parser.add_argument('--tmpdir', default='./tmp', help='临时文件目录')
    # 对抗攻击参数
    parser.add_argument('--attack', action='store_true', help='是否启用对抗攻击')
    parser.add_argument('--attack_method', default='pgd', help='攻击方法（仅支持pgd）')
    parser.add_argument('--epsilon', type=float, default=8/255, help='扰动上限（像素值0-255）')
    parser.add_argument('--step_size', type=float, default=2/255, help='PGD迭代步长')
    parser.add_argument('--max_iterations', type=int, default=10, help='PGD迭代次数')
    parser.add_argument('--random_start', action='store_true', help='PGD随机初始化扰动')
    parser.add_argument('--gen_adv_sample_num', type=int, default=10, help='生成对抗样本数量')
    parser.add_argument('--loss_target', default='traj', help='攻击目标损失 (det/traj/map)')
    # 设备参数
    parser.add_argument('--gpu-id', type=int, default=2, help='指定GPU编号（第三张卡为2）')
    args = parser.parse_args()

    # 2. 设置GPU设备
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.cuda.set_device(args.gpu_id)

    # 3. 加载VAD配置文件
    cfg = Config.fromfile(args.config)
    distributed = False
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.dist_params)
        distributed = True

    # 4. 构建VAD测试数据集和数据加载器
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=getattr(cfg.data, 'nonshuffler_sampler', None),
    )

    # 5. 构建VAD模型并加载权重
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 半精度推理（减少显存占用）
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # 加载权重（抑制冗余输出）
    with suppress_stdout():
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    # 模型移到指定GPU
    model = model.cuda(args.gpu_id)

    # 6. 执行对抗攻击（若启用）
    if args.attack:
        # 初始化攻击器
        attacker = VADAttacker(
            args=args,
            cfg=cfg,
            model=model,
            dataset=dataset,
            dataloader=data_loader
        )
        # 生成对抗样本
        attacked_data = attacker.run_attack()
        # 替换原始数据加载器（简化版：直接使用攻击后的批次列表）
        data_loader = attacked_data

    # 7. 模型推理与评估（适配攻击后的数据）
    print("\n开始模型推理与评估...")
    if isinstance(data_loader, list):
        # 攻击后的数据是批次列表，逐批次推理
        outputs = []
        for batch in data_loader:
            with torch.no_grad():
                result = model(return_loss=False, **batch) if isinstance(batch, dict) else model(batch, return_loss=False)
                outputs.append(result)
    else:
        # 原始数据加载器，使用单卡测试接口
        model.eval()
        outputs = single_gpu_test(model, data_loader)

    # 8. 评估并打印结果
    eval_kwargs = dict(metric=args.eval, tmpdir=args.tmpdir)
    eval_results = dataset.evaluate(outputs, **eval_kwargs)
    print("\n===== 评估结果 =====")
    for k, v in eval_results.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()