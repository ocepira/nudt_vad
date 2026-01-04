# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import sys
sys.path.append('')
import numpy as np
import argparse
import mmcv
import os
import copy
import torch
from easydict import EasyDict
import yaml  # 添加缺失的yaml导入
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
from utils.vadattack import ImageAttacker
# mmcv.disable_progressbar()

def sse_print(event: str, data: dict) -> str:
    """
    SSE 打印
    :param event: 事件名称
    :param data: 事件数据（字典或能被 json 序列化的对象）
    :return: SSE 格式字符串
    """
    # 处理数据，确保它可以被JSON序列化
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        else:
            # 对于其他不可序列化的对象，转换为字符串表示
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    # 将数据转成 JSON 字符串
    try:
        cleaned_data = convert_for_json(data)
        json_str = json.dumps(cleaned_data, ensure_ascii=False)
    except Exception as e:
        # 如果仍然失败，则只发送简单的错误消息
        json_str = json.dumps({"error": "Failed to serialize data", "exception": str(e)}, ensure_ascii=False)
    
    # 按 SSE 协议格式拼接
    message = f"event: {event}\n" \
              f"data: {json_str}\n"
    print(message, flush=True)
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
    # 修改为可选参数
    parser.add_argument('config', nargs='?', help='test config file path')
    parser.add_argument('checkpoint', nargs='?', help='checkpoint file')
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
    # parser.add_argument('--attack', action='store_true', help='是否启用对抗攻击')
    # parser.add_argument('--attack-method', type=str, default='fgsm', 
    #                    choices=['fgsm', 'pgd', 'cw', 'bim', 'deepfool'], 
    #                    help='对抗攻击方法')
    parser.add_argument('--steps', type=int, default=10, help='攻击迭代次数(PGD/BIM)')
    parser.add_argument('--alpha', type=float, default=2/255, help='攻击步长(PGD/BIM)')
    parser.add_argument('--cfg-yaml', type=str, help='攻击配置文件路径')
    parser.add_argument('--process', type=str, default='attack', help='process type: train/test',choices=['test', 'attack'])
   ##攻击
    parser.add_argument('--image-path', type=str, help='输入图像路径')
    parser.add_argument('--attack-method', type=str, default='deepfool', 
                        choices=['fgsm', 'pgd', 'bim','badnet', 'squareattack', 'nes'], 
                        help='对抗攻击方法')
    parser.add_argument('--epsilon', type=float, default=8/255, help='扰动强度')
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

    ## 防御

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
def load_yaml(load_path):
    with open(load_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
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
    args = parse_args()  # 提前调用parse_args()
    # 添加检查：当 process 为 test 时，config 和 checkpoint 是必需的
    if args.process == "test":
        if not args.config or not args.checkpoint:
            print("Error: the following arguments are required: config, checkpoint")
            sys.exit(1)
    # 添加检查：当 process 为 attack 时，image-path 是必需的
    if args.process == "attack" and not args.image_path:
        print("Error: the following arguments are required: --image-path")
        sys.exit(1)
    
    if args.process == "test":  # 修复变量名 arg -> args
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

        if not distributed:
            # assert False
            model = MMDataParallel(model, device_ids=[0])
            with suppress_stdout_stderr():
                outputs = single_gpu_test(model, data_loader, args.show, args.show_dir) 
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            with suppress_stdout_stderr():
                outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                                args.gpu_collect)

        tmp = {}
        tmp['bbox_results'] = outputs
        outputs = tmp
        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                # assert False
                if isinstance(outputs, list):
                    mmcv.dump(outputs, args.out)
                else:
                    mmcv.dump(outputs['bbox_results'], args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
                '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
            if args.format_only:
                dataset.format_results(outputs['bbox_results'], **kwargs)

            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))

                print(dataset.evaluate(outputs['bbox_results'], **eval_kwargs))
        
            # # # NOTE: record to json
            # json_path = args.json_dir
            # if not os.path.exists(json_path):
            #     os.makedirs(json_path)
            
            # metric_all = []
            # for res in outputs['bbox_results']:
            #     for k in res['metric_results'].keys():
            #         if type(res['metric_results'][k]) is np.ndarray:
            #             res['metric_results'][k] = res['metric_results'][k].tolist()
            #     metric_all.append(res['metric_results'])
            
            # print('start saving to json done')
            # with open(json_path+'/metric_record.json', "w", encoding="utf-8") as f2:
            #     json.dump(metric_all, f2, indent=4)
            # print('save to json done')
        # if args.attack:
        #     attack_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        #     attacked_data_loader = 
        #     data_loader = data_loader
        #     m
        #     attacker = ImageAttacker(attack_model, method=args.attack_method, epsilon=args.epsilon,
    elif args.process == "attack":
            # 获取设备信息
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            attacker = ImageAttacker(
                # model_name="Standard",
                # dataset=args.dataset,
                attack_method=args.attack_method,
                eps=args.epsilon,
                alpha=args.alpha,
                steps=args.steps,
                device=device
            )
            # attacker.attack(
            #     image_path=args.image_path,
            #     save_path=args.save_path
            # )
            # 加载模型
            # sse_print("model_loading", {"message": "正在加载模型..."})
            if args.dataset.lower() == 'cifar10':
                try:
                    from robustbench.utils import load_model
                    model = load_model(model_name = 'Standard', norm='Linf', dataset=args.dataset).to(device)
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

if __name__ == '__main__':
    args = parse_args()
    sse_input_path_validated(args)
    # sse_output_path_validated(args)
    # print(args.checkpoint)
    main()