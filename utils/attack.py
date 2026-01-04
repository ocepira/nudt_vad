from typing import Any, Dict
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
import glob
import yaml
from easydict import EasyDict


# from ultralytics import YOLO


# from ultralytics.data.utils import check_cls_dataset, check_det_dataset
# from ultralytics.data import build_yolo_dataset, ClassificationDataset_nudt, build_dataloader
# from ultralytics.utils import TQDM, emojis
# from ultralytics.utils.torch_utils import select_device

# from ultralytics.models.yolo.classify.predict import ClassificationPredictor
# from ultralytics.nn.tasks import ClassificationModel, DetectionModel
# from ultralytics.nn.tasks import torch_safe_load
# from ultralytics.utils.loss import v8DetectionLoss_nudt

# from nudt_ultralytics.callbacks.callbacks import callbacks_dict
# from ultralytics.utils import callbacks

# from utils.sse import sse_adv_samples_gen_validated, sse_model_loaded
from utils.yaml_rw import load_yaml

import sys
sys.path.append('')
import numpy as np
import argparse
import mmcv
import os
import copy
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
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
# from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.VAD.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import json
import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('mmcv').setLevel(logging.WARNING)
logging.getLogger('mmcv').setLevel(logging.WARNING)
logging.getLogger('mmcv.runner').setLevel(logging.WARNING)
from utils.attack import attacks

import sys
from contextlib import contextmanager
from contextlib import contextmanager
import sys
from io import StringIO



class attacks:
    def __init__(self, args):
        self.args = args
        self.cfg = Config.fromfile(args.cfg_yaml)
        self.device = self.cfg.device
        
        if self.cfg.task == "vad":
            self.cfg.model.pretrained = None 
            self.cfg = Config.fromfile(args.config)
            if isinstance(self.cfg.data.test, dict):
                self.cfg.data.test.test_mode = True
                samples_per_gpu = self.cfg.data.test.pop('samples_per_gpu', 1)
                if samples_per_gpu > 1:
                    # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                    self.cfg.data.test.pipeline = replace_ImageToTensor(
                        self.cfg.data.test.pipeline)
            elif isinstance(self.cfg.data.test, list):
                for ds_cfg in self.cfg.data.test:
                    ds_cfg.test_mode = True
                samples_per_gpu = max(
                    [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in self.cfg.data.test])
                if samples_per_gpu > 1:
                    for ds_cfg in self.cfg.data.test:
                        ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
            self.dataset = build_dataset(self.cfg.data.test)
            self.dataloader = build_dataloader(
                self.dataset,
                samples_per_gpu=samples_per_gpu,
                workers_per_gpu=self.cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False,
                nonshuffler_sampler=self.cfg.data.nonshuffler_sampler if hasattr(self.cfg.data, 'nonshuffler_sampler') else None,
            )
            self.cfg.model.train_cfg = None  # 清除训练配置
            self.model = build_model(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
            self.checkpoint = load_checkpoint(self.model, args.checkpoint, map_location='cpu')

    def get_multiview_images(self, batch):
        """
        从批次数据中提取多视角图像
        """
        # 如果数据集中包含多视角图像信息
        if 'img' in batch and 'img_shape' in batch:
            # nuScenes数据集通常有多个摄像头视角
            # 图像数据形状通常是 [batch_size, num_cameras, channels, height, width]
            multiview_images = batch['img']
            return multiview_images
        return None

    def process_single_camera_attack(self, batch, camera_index=0):
        """
        处理单个摄像头视角的攻击
        """
        # 提取特定摄像头的图像
        if 'img' in batch:
            if len(batch['img'].shape) == 5:  # 多视角图像 [B, N, C, H, W]
                single_view_image = batch['img'][:, camera_index, :, :, :]
            else:  # 单视角图像 [B, C, H, W]
                single_view_image = batch['img']
            return single_view_image
        return None

            
        
    def save_vad_image(self, adv_image, ori_image_file):
        return 0

    
    def run_adv(self):
        for batch_i, batch in enumerate(self.dataloader):
            # print(batch.keys()) # dict_keys(['batch_idx', 'bboxes', 'cls', 'im_file', 'img', 'ori_shape', 'ratio_pad', 'resized_shape'])
            if self.args.attack_method == 'pgd':
                loss_function = F.l1_loss
                adv_images = self.pgd(batch, eps=self.args.epsilon, alpha=self.args.step_size, steps=self.args.max_iterations, random_start=self.args.random_start, loss_function=loss_function)
                
            # elif self.args.attack_method == 'fgsm':
            #     adv_images = self.fgsm(batch, eps=self.args.epsilon, loss_function=self.args.loss_function)
            # elif self.args.attack_method == 'cw':
            #     adv_images = self.cw(batch, c=1, kappa=0, steps=self.args.max_iterations, lr=self.args.lr, optimization_method=self.args.optimization_method)
            # elif self.args.attack_method == 'bim':
            #     adv_images = self.bim(batch, eps=self.args.epsilon, alpha=self.args.step_size, steps=self.args.max_iterations, loss_function=self.args.loss_function)
            # elif self.args.attack_method == 'deepfool':
            #     adv_images, _ = self.deepfool(batch, steps=self.args.max_iterations, overshoot=0.02)
            # else:
            #     raise ValueError('Invalid attach method!')
            
            # if self.cfg.task == "detect":
            #     self.detect_save_adv_image(adv_image=adv_images[0], ori_image_file=batch["im_file"][0])
            # elif self.cfg.task == "classify":
            #     self.classify_save_adv_image(adv_image=adv_images[0], ori_image_file=batch["im_file"][0])
            # if batch_i == self.args.gen_adv_sample_num - 1:
            #     break

        
    # def gen_loss_fn(self, name):
    #     loss_fn = name.lower()
    #     if loss_fn == 'cross_entropy':
    #         loss_function = F.cross_entropy
    #     elif loss_fn == 'mse':
    #         loss_function = F.mse_loss
    #     elif loss_fn == 'l1':
    #         loss_function = F.l1_loss
    #     elif loss_fn == 'binary_cross_entropy':
    #         loss_function = F.binary_cross_entropy
    #     else:
    #         raise ValueError("Invalid Loss Type!")
    #     return loss_function
    
    def get_optimizer(self, name, param, lr):
        optimization_method = name.lower()
        if optimization_method == 'sgd':
            optimizer = torch.optim.SGD(param, lr=lr)
        elif self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(param, lr=lr)
        else:
            raise ValueError('Invalid optimizer type!')
###################################################################################################################################################
    

    def pgd(self, batch, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, loss_function='cross_entropy'):
        '''
        PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
        [https://arxiv.org/abs/1706.06083]

        Distance Measure : Linf
    
        Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''

        # if self.cfg.task == 'classify':
        #     images = self.classify_preprocess(im=batch["img"])
        #     labels = batch["cls"]
        #     loss_fn = self.gen_loss_fn(loss_function)
        # elif self.cfg.task == 'detect':
        #     images = self.detect_preprocess(im=batch["img"])
        #     loss_fn = v8DetectionLoss_nudt(self.model, self.cfg)
        # adv_images = images.clone().detach()
        if len(images.shape) == 5:  # [B, N, C, H, W] 多视角图像
            batch_size, num_cams, channels, height, width = images.shape
            # 可以选择处理所有视角或特定视角
            # 这里展示处理第一个摄像头视角的方法
            images = images[:, 0, :, :, :]  # 选择前视摄像头 [B, C, H, W]
            
        adv_images = images.clone().detach()

        if random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        for _ in range(steps):
            adv_images.requires_grad = True
            if self.cfg.task == 'classify':
                preds = self.model.predict(x=adv_images)
                loss = loss_fn(preds, labels)
            else:
                batch['img'] = adv_images
                preds = self.model.forward(x=batch["img"])
                loss, loss_items = loss_fn(preds, batch) # loss[0]: box, loss[1]: cls, loss[2]: df1
                # loss = loss.sum()
                loss = loss[1]
            
            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
    
    
    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))
    
    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    # f-function in the paper
    def f(self, outputs, labels, kappa):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs.to(self.device), dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs.to(self.device), dim=1)[0]

        return torch.clamp((real - other), min=-kappa)

    
    def _forward_indiv(self, image, label):
        image.requires_grad = True
        
        if self.cfg.task == 'classify':
            preds = self.model.predict(x=image)
        else:
            raise ValueError('Unsupported attach method!')
        
        fs = preds.to(self.device)
        _, pre = torch.max(fs, dim=-1)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (
            torch.abs(f_prime[hat_L])
            * w_prime[hat_L]
            / (torch.norm(w_prime[hat_L], p=2) ** 2)
        )

        target_label = hat_L if hat_L < label else hat_L + 1

        adv_image = image + (1 + overshoot) * delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
    
    # def loss_single(self,
    #                 cls_scores,
    #                 bbox_preds,
    #                 traj_preds,
    #                 traj_cls_preds,
    #                 gt_bboxes_list,
    #                 gt_labels_list,
    #                 gt_attr_labels_list,
    #                 gt_bboxes_ignore_list=None):
    #     """"Loss function for outputs from a single decoder layer of a single
    #     feature level.
    #     Args:
    #         cls_scores (Tensor): Box score logits from a single decoder layer
    #             for all images. Shape [bs, num_query, cls_out_channels].
    #         bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
    #             for all images, with normalized coordinate (cx, cy, w, h) and
    #             shape [bs, num_query, 4].
    #         gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
    #             with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels_list (list[Tensor]): Ground truth class indices for each
    #             image with shape (num_gts, ).
    #         gt_bboxes_ignore_list (list[Tensor], optional): Bounding
    #             boxes which can be ignored for each image. Default None.
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components for outputs from
    #             a single decoder layer.
    #     """
    #     num_imgs = cls_scores.size(0)
    #     cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    #     bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    #     cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
    #                                        gt_bboxes_list, gt_labels_list,
    #                                        gt_attr_labels_list, gt_bboxes_ignore_list)

    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      traj_targets_list, traj_weights_list, gt_fut_masks_list,
    #      num_total_pos, num_total_neg) = cls_reg_targets

    #     labels = torch.cat(labels_list, 0)
    #     label_weights = torch.cat(label_weights_list, 0)
    #     bbox_targets = torch.cat(bbox_targets_list, 0)
    #     bbox_weights = torch.cat(bbox_weights_list, 0)
    #     traj_targets = torch.cat(traj_targets_list, 0)
    #     traj_weights = torch.cat(traj_weights_list, 0)
    #     gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

    #     # classification loss
    #     cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
    #     # construct weighted avg_factor to match with the official DETR repo
    #     cls_avg_factor = num_total_pos * 1.0 + \
    #         num_total_neg * self.bg_cls_weight
    #     if self.sync_cls_avg_factor:
    #         cls_avg_factor = reduce_mean(
    #             cls_scores.new_tensor([cls_avg_factor]))

    #     cls_avg_factor = max(cls_avg_factor, 1)
    #     loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    #     # Compute the average number of gt boxes accross all gpus, for
    #     # normalization purposes
    #     num_total_pos = loss_cls.new_tensor([num_total_pos])
    #     num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

    #     # regression L1 loss
    #     bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
    #     normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
    #     isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
    #     bbox_weights = bbox_weights * self.code_weights
    #     loss_bbox = self.loss_bbox(
    #         bbox_preds[isnotnan, :10],
    #         normalized_bbox_targets[isnotnan, :10],
    #         bbox_weights[isnotnan, :10],
    #         avg_factor=num_total_pos)

    #     # traj regression loss
    #     best_traj_preds = self.get_best_fut_preds(
    #         traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
    #         traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

    #     neg_inds = (bbox_weights[:, 0] == 0)
    #     traj_labels = self.get_traj_cls_target(
    #         traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
    #         traj_targets.reshape(-1, self.fut_ts, 2),
    #         gt_fut_masks, neg_inds)

    #     loss_traj = self.loss_traj(
    #         best_traj_preds[isnotnan],
    #         traj_targets[isnotnan],
    #         traj_weights[isnotnan],
    #         avg_factor=num_total_pos)

    #     if self.use_traj_lr_warmup:
    #         loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
    #         loss_traj = loss_scale_factor * loss_traj

    #     # traj classification loss
    #     traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
    #     # construct weighted avg_factor to match with the official DETR repo
    #     traj_cls_avg_factor = num_total_pos * 1.0 + \
    #         num_total_neg * self.traj_bg_cls_weight
    #     if self.sync_cls_avg_factor:
    #         traj_cls_avg_factor = reduce_mean(
    #             traj_cls_scores.new_tensor([traj_cls_avg_factor]))

    #     traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
    #     loss_traj_cls = self.loss_traj_cls(
    #         traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
    #     )

    #     if digit_version(TORCH_VERSION) >= digit_version('1.8'):
    #         loss_cls = torch.nan_to_num(loss_cls)
    #         loss_bbox = torch.nan_to_num(loss_bbox)
    #         loss_traj = torch.nan_to_num(loss_traj)
    #         loss_traj_cls = torch.nan_to_num(loss_traj_cls)

    #     return loss_cls, loss_bbox, loss_traj, loss_traj_cls
    