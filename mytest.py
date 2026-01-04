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
# from utils.attack import attacks

import sys
from contextlib import contextmanager
from contextlib import contextmanager
import sys
from io import StringIO
from utils.sse import sse_input_path_validated,sse_output_path_validated

# 导入VADAttacker类
from utils.myattack import VADAttacker
@contextmanager
def suppress_stdout():
    """临时屏蔽标准输出的上下文管理器"""
    original_stdout = sys.stdout  # 保存原始stdout
    sys.stdout = open(os.devnull, 'w')  # 重定向到空设备
    try:
        yield
    finally:
        sys.stdout.close()  # 关闭空设备文件
        sys.stdout = original_stdout  # 恢复原始stdout
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--json_dir', help='json parent dir name file') # NOTE: json file parent folder name
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--input_path', type=str, default='./data/nuscenes', help='input path')
    parser.add_argument('--ouput_path', type=str, default='../output', help='output path')
    # 2025/12/9 add attack args
    parser.add_argument('--attack', action='store_true', help='是否启用对抗攻击')
    parser.add_argument('--attack-method', type=str, default='pgd', 
                       choices=['fgsm', 'pgd', 'cw', 'bim', 'deepfool'], 
                       help='对抗攻击方法')
    parser.add_argument('--epsilon', type=float, default=8/255, help='扰动强度')
    parser.add_argument('--steps', type=int, default=10, help='攻击迭代次数(PGD/BIM)')
    parser.add_argument('--alpha', type=float, default=2/255, help='攻击步长(PGD/BIM)')
    parser.add_argument('--cfg-yaml', type=str, help='攻击配置文件路径')
    parser.add_argument('--gen_adv_sample_num', type=int, default=10, help='生成对抗样本数量')
    parser.add_argument('--step_size', type=float, default=2/255, help='PGD攻击步长')
    parser.add_argument('--max_iterations', type=int, default=50, help='最大迭代次数')
    parser.add_argument('--random_start', action='store_true', help='随机初始化')
    parser.add_argument('--loss_target', type=str, default='cross_entropy', choices=['cross_entropy', 'mse', 'l1'], help='目标损失类型')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

@contextmanager
def suppress_stdout_stderr():
    """同时抑制 stdout、stderr 和文件描述符"""
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    
    # 保存原始文件描述符
    save_stdout_fd = os.dup(1)
    save_stderr_fd = os.dup(2)
    
    try:
        # 重定向到 /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        
        # Python 层面也要重定向
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        yield
    finally:
        # 恢复文件描述符
        os.dup2(save_stdout_fd, 1)
        os.dup2(save_stderr_fd, 2)
        os.close(save_stdout_fd)
        os.close(save_stderr_fd)
        
        # 恢复 sys.stdout 和 sys.stderr
        sys.stdout = save_stdout
        sys.stderr = save_stderr

def main():
    args = parse_args()
    # mmcv.disable_progressbar()
    
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                # print(_module_path)  已屏蔽
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                # print(_module_path)  已屏蔽
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    print("=" * 50)
    print("数据集信息:")
    if hasattr(dataset, 'data_root'):
        print(f"数据根目录: {dataset.data_root}")
    if hasattr(dataset, 'ann_file'):
        print(f"标注文件: {dataset.ann_file}")
    if hasattr(dataset, 'img_prefix'):
        print(f"图像前缀: {dataset.img_prefix}")
    
    # 如果是ConcatDataset，打印每个子数据集的信息
    if hasattr(dataset, 'datasets'):
        print(f"包含 {len(dataset.datasets)} 个子数据集:")
        for i, sub_dataset in enumerate(dataset.datasets):
            print(f"  子数据集 {i+1}:")
            if hasattr(sub_dataset, 'data_root'):
                print(f"    数据根目录: {sub_dataset.data_root}")
            if hasattr(sub_dataset, 'ann_file'):
                print(f"    标注文件: {sub_dataset.ann_file}")
            if hasattr(sub_dataset, 'img_prefix'):
                print(f"    图像前缀: {sub_dataset.img_prefix}")
    print("=" * 50)
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # print("mmcv.runner 日志级别:", logging.getLogger('mmcv.runner').getEffectiveLevel()) 
    with suppress_stdout():
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    # 对抗攻击处理
    if args.attack:
        print("启用对抗攻击...")
        # 创建攻击器实例
        attacker = VADAttacker(
            args=args,
            cfg=cfg,
            model=model,
            dataset=dataset,
            dataloader=data_loader
        )
        # 生成对抗样本
        attacked_data = attacker.run_attack()
        # 使用对抗样本进行测试
        data_loader = attacked_data
        print("对抗攻击完成，开始使用对抗样本进行测试...")

    if not distributed:
        # assert False
        model = MMDataParallel(model, device_ids=[0])
        with suppress_stdout_stderr():
            if args.attack and isinstance(data_loader, list):
                # 处理对抗攻击后的数据（批次列表）
                outputs = []
                for batch in data_loader:
                    with torch.no_grad():
                        result = model(return_loss=False, rescale=True, **batch)
                        outputs.append(result)
            else:
                # 原始数据加载器
                outputs = single_gpu_test(model, data_loader, args.show, args.show_dir) 
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        with suppress_stdout_stderr():
            outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                            args.gpu_collect)

    # 修复：确保outputs是字典格式，无论是否使用对抗攻击
    # tmp = {}
    # tmp['bbox_results'] = outputs
    # outputs = tmp
    # rank, _ = get_dist_info()
    # if rank == 0:
    #     if args.out:
    #         print(f'\nwriting results to {args.out}')
    #         # assert False
    #         if isinstance(outputs, list):
    #             mmcv.dump(outputs, args.out)
    #         else:
    #             mmcv.dump(outputs['bbox_results'], args.out)
    #     kwargs = {} if args.eval_options is None else args.eval_options
    #     kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
    #         '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
    #     if args.format_only:
    #         dataset.format_results(outputs['bbox_results'], **kwargs)

    #     if args.eval:
    #         eval_kwargs = cfg.get('evaluation', {}).copy()
    #         # hard-code way to remove EvalHook args
    #         for key in [
    #                 'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
    #                 'rule'
    #         ]:
    #             eval_kwargs.pop(key, None)
    #         eval_kwargs.update(dict(metric=args.eval, **kwargs))

    #         print(dataset.evaluate(outputs['bbox_results'], **eval_kwargs))
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)  # 直接保存原生outputs
    
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join(
            'test', 
            args.config.split('/')[-1].split('.')[-2], 
            time.ctime().replace(' ', '_').replace(':', '_')
        )
    
    if args.format_only:
        dataset.format_results(outputs, **kwargs)  # 直接传outputs

    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # 移除EvalHook无关参数
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))

        # 核心：直接传入原生outputs
        eval_results = dataset.evaluate(outputs, **eval_kwargs)
        print("\n===== 对抗攻击后评估结果 =====")
        for k, v in eval_results.items():
            print(f"{k}: {v}")
    if args.attack:
        print("启用对抗攻击...")
        # 创建攻击器实例
        attacker = VADAttacker(
            args=args,
            cfg=cfg,
            model=model,
            dataset=dataset,
            dataloader=data_loader
        )
        # 生成对抗样本（返回包装的DataLoader）
        attacked_dataloader = attacker.run_attack()
        # 替换为对抗样本DataLoader
        data_loader = attacked_dataloader
        print("对抗攻击完成，开始使用对抗样本进行测试...")

# 统一使用single_gpu_test推理（保证输出格式一致）
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        with suppress_stdout_stderr():
            # 无论是否攻击，都用single_gpu_test（关键！）
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        with suppress_stdout_stderr():
            outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    
if __name__ == '__main__':
    args = parse_args()
    sse_input_path_validated(args)
    # sse_output_path_validated(args)
    # print(args.checkpoint)
    main()