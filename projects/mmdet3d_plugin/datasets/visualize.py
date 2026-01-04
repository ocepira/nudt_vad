import sys
sys.path.append('')
import os
import argparse
import os.path as osp
from PIL import Image
from tqdm import tqdm
from typing import List, Dict

import cv2
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from mmdet.datasets.pipelines import to_tensor
from matplotlib.collections import LineCollection
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

from projects.mmdet3d_plugin.core.bbox.structures.nuscenes_box import CustomNuscenesBox, CustomDetectionBox, color_map
from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import VectorizedLocalMap, LiDARInstanceLines

class VADNuScenesVisualizer:
    """
    VAD模型推理结果可视化类 - 封装所有可视化功能
    支持：BEV视角预测可视化、6相机画面标注、轨迹绘制、规划指令显示、自动生成mp4视频
    """
    def __init__(self, result_path, save_path, nusc_version='v1.0-mini', nusc_dataroot=r'/data6/user24215461/autodrive/VAD/data/nuscenes'):
        """
        初始化可视化器
        :param result_path: 推理结果文件路径 (.pkl格式)
        :param save_path: 可视化结果保存目录
        :param nusc_version: nuScenes数据集版本
        :param nusc_dataroot: nuScenes数据集根路径
        """
        self.result_path = result_path
        self.save_path = save_path
        self.nusc_version = nusc_version
        self.nusc_dataroot = nusc_dataroot
        
        # 初始化数据集对象+加载推理结果
        self.nusc = NuScenes(version=self.nusc_version, dataroot=self.nusc_dataroot, verbose=True)
        self.bevformer_results = mmcv.load(self.result_path)
        self.sample_token_list = list(self.bevformer_results['results'].keys())
        
        # 初始化视频写入器参数
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.video_fps = 10
        self.video_size = (2933, 800)
        self.video_path = osp.join(self.save_path, 'vis.mp4')
        self.video = cv2.VideoWriter(self.video_path, self.fourcc, self.video_fps, self.video_size, True)
        
        # 固定参数
        self.cams = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
        self.cmd_list = ['Turn Right', 'Turn Left', 'Go Straight']
        self.conf_th = 0.4
        self.colors_plt = ['cornflowerblue', 'royalblue', 'slategrey']

    def render_annotation(self, anntoken: str, margin: float = 10, view: np.ndarray = np.eye(4),
                          box_vis_level: BoxVisibility = BoxVisibility.ANY, out_path: str = 'render.png', extra_info: bool = False) -> None:
        ann_record = self.nusc.get('sample_annotation', anntoken)
        sample_record = self.nusc.get('sample', ann_record['sample_token'])
        assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

        boxes, cam = [], []
        cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
        all_bboxes = []
        select_cams = []
        for cam in cams:
            _, boxes, _ = self.nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level, selected_anntokens=[anntoken])
            if len(boxes) > 0:
                all_bboxes.append(boxes)
                select_cams.append(cam)

        num_cam = len(all_bboxes)
        fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
        select_cams = [sample_record['data'][cam] for cam in select_cams]

        lidar = sample_record['data']['LIDAR_TOP']
        data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
        LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
        for box in boxes:
            c = np.array(self.get_color(box.name)) / 255.0
            box.render(axes[0], view=view, colors=(c, c, c))
            corners = view_points(boxes[0].corners(), view, False)[:2, :]
            axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
            axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
            axes[0].axis('off')
            axes[0].set_aspect('equal')

        for i in range(1, num_cam + 1):
            cam = select_cams[i - 1]
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(cam, selected_anntokens=[anntoken])
            im = Image.open(data_path)
            axes[i].imshow(im)
            axes[i].set_title(self.nusc.get('sample_data', cam)['channel'])
            axes[i].axis('off')
            axes[i].set_aspect('equal')
            for box in boxes:
                c = np.array(self.get_color(box.name)) / 255.0
                box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))
            axes[i].set_xlim(0, im.size[0])
            axes[i].set_ylim(im.size[1], 0)

        if extra_info:
            rcParams['font.family'] = 'monospace'
            w, l, h = ann_record['size']
            category = ann_record['category_name']
            lidar_points = ann_record['num_lidar_pts']
            radar_points = ann_record['num_radar_pts']
            sample_data_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
            pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
            dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))
            information = ' \n'.join(['category: {}'.format(category),'', '# lidar points: {0:>4}'.format(lidar_points),
                                      '# radar points: {0:>4}'.format(radar_points),'', 'distance: {:>7.3f}m'.format(dist),
                                      '', 'width:  {:>7.3f}m'.format(w), 'length: {:>7.3f}m'.format(l), 'height: {:>7.3f}m'.format(h)])
            plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

        if out_path is not None:
            plt.savefig(out_path)
        plt.close()

    def get_sample_data(self, sample_data_token: str, box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        selected_anntokens=None, use_flat_vehicle_coordinates: bool = False):
        sd_record = self.nusc.get('sample_data', sample_data_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.nusc.get('sensor', cs_record['sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        data_path = self.nusc.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        if selected_anntokens is not None:
            boxes = list(map(self.nusc.get_box, selected_anntokens))
        else:
            boxes = self.nusc.get_boxes(sample_data_token)

        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue
            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def get_predicted_data(self, sample_data_token: str, box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           selected_anntokens=None, use_flat_vehicle_coordinates: bool = False, pred_anns=None):
        sd_record = self.nusc.get('sample_data', sample_data_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.nusc.get('sensor', cs_record['sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        data_path = self.nusc.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        boxes = pred_anns
        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue
            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def lidiar_render(self, sample_token, traj_use_perstep_offset=True):
        bbox_gt_list = []
        bbox_pred_list = []
        sample_rec = self.nusc.get('sample', sample_token)
        anns = sample_rec['anns']
        sd_record = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

        for ann in anns:
            content = self.nusc.get('sample_annotation', ann)
            gt_fut_trajs, gt_fut_masks = self.get_gt_fut_trajs(anno=content, cs_record=cs_record, pose_record=pose_record, fut_ts=6)
            try:
                bbox_gt_list.append(CustomDetectionBox(
                    sample_token=content['sample_token'],translation=tuple(content['translation']),size=tuple(content['size']),
                    rotation=tuple(content['rotation']),velocity=self.nusc.box_velocity(content['token'])[:2],
                    fut_trajs=tuple(gt_fut_trajs),ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content else tuple(content['ego_translation']),
                    num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                    detection_name=category_to_detection_name(content['category_name']),
                    detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                    attribute_name=''))
            except:
                pass

        bbox_anns = self.bevformer_results['results'][sample_token]
        for content in bbox_anns:
            bbox_pred_list.append(CustomDetectionBox(
                sample_token=content['sample_token'],translation=tuple(content['translation']),size=tuple(content['size']),
                rotation=tuple(content['rotation']),velocity=tuple(content['velocity']),fut_trajs=tuple(content['fut_traj']),
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=content['detection_name'],detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=content['attribute_name']))
                
        gt_annotations = EvalBoxes()
        pred_annotations = EvalBoxes()
        gt_annotations.add_boxes(sample_token, bbox_gt_list)
        pred_annotations.add_boxes(sample_token, bbox_pred_list)
        self.visualize_sample(sample_token, gt_annotations, pred_annotations, traj_use_perstep_offset=traj_use_perstep_offset)

    def get_color(self, category_name: str):
        a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
         'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
         'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
         'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
         'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
         'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
         'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation', 'vehicle.ego']
        class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier','motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        if category_name == 'bicycle':
            return self.nusc.colormap['vehicle.bicycle']
        elif category_name == 'construction_vehicle':
            return self.nusc.colormap['vehicle.construction']
        elif category_name == 'traffic_cone':
            return self.nusc.colormap['movable_object.trafficcone']

        for key in self.nusc.colormap.keys():
            if category_name in key:
                return self.nusc.colormap[key]
        return [0, 0, 0]

    def boxes_to_sensor(self, boxes: List[EvalBox], pose_record: Dict, cs_record: Dict):
        boxes_out = []
        for box in boxes:
            box = CustomNuscenesBox(box.translation, box.size, Quaternion(box.rotation), box.fut_trajs, name=box.detection_name)
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)
            boxes_out.append(box)
        return boxes_out

    def get_gt_fut_trajs(self, anno, cs_record, pose_record, fut_ts) -> None:
        box = Box(anno['translation'], anno['size'], Quaternion(anno['rotation']))
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        
        gt_fut_trajs = np.zeros((fut_ts, 2))
        gt_fut_masks = np.zeros((fut_ts))
        gt_fut_trajs[:] = box.center[:2]
        cur_box = box
        cur_anno = anno
        for i in range(fut_ts):
            if cur_anno['next'] != '':
                anno_next = self.nusc.get('sample_annotation', cur_anno['next'])
                box_next = Box(anno_next['translation'], anno_next['size'], Quaternion(anno_next['rotation']))
                box_next.translate(-np.array(pose_record['translation']))
                box_next.rotate(Quaternion(pose_record['rotation']).inverse)
                box_next.translate(-np.array(cs_record['translation']))
                box_next.rotate(Quaternion(cs_record['rotation']).inverse)
                gt_fut_trajs[i] = box_next.center[:2] - cur_box.center[:2]
                gt_fut_masks[i] = 1
                cur_anno = anno_next
                cur_box = box_next
            else:
                gt_fut_trajs[i:] = 0
                break         
        return gt_fut_trajs.reshape(-1).tolist(), gt_fut_masks.reshape(-1).tolist()

    def get_gt_vec_maps(self, sample_token, data_root='data/nuscenes/', pc_range=[-15.0, -30.0, -4.0, 15.0, 30.0, 4.0],
                        padding_value=-10000, map_classes=['divider', 'ped_crossing', 'boundary'], map_fixed_ptsnum_per_line=20) -> None:
        sample_rec = self.nusc.get('sample', sample_token)
        sd_record = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(cs_record['rotation']).rotation_matrix
        lidar2ego[:3, 3] = cs_record['translation']
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(pose_record['rotation']).rotation_matrix
        ego2global[:3, 3] = pose_record['translation']
        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        patch_h = pc_range[4]-pc_range[1]
        patch_w = pc_range[3]-pc_range[0]
        patch_size = (patch_h, patch_w)

        vector_map = VectorizedLocalMap(data_root, patch_size=patch_size,map_classes=map_classes, 
                                        fixed_ptsnum_per_line=map_fixed_ptsnum_per_line,padding_value=padding_value)
        anns_results = vector_map.gen_vectorized_samples(map_location=self.nusc.get('log', self.nusc.get('scene', sample_rec['scene_token'])['log_token'])['location'],
                                                         lidar2global_translation=lidar2global_translation, lidar2global_rotation=lidar2global_rotation)
        
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                gt_vecs_pts_loc = gt_vecs_pts_loc
        return gt_vecs_pts_loc, gt_vecs_label

    def visualize_sample(self, sample_token: str, gt_boxes: EvalBoxes, pred_boxes: EvalBoxes,
                         nsweeps: int = 1, pc_range: list = [-30.0, -30.0, -4.0, 30.0, 30.0, 4.0], verbose: bool = True,
                         traj_use_perstep_offset: bool = True, data_root='data/nuscenes/') -> None:
        sample_rec = self.nusc.get('sample', sample_token)
        sd_record = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        
        boxes_gt_global = gt_boxes[sample_token]
        boxes_est_global = pred_boxes[sample_token]
        boxes_gt = self.boxes_to_sensor(boxes_gt_global, pose_record, cs_record)
        boxes_est = self.boxes_to_sensor(boxes_est_global, pose_record, cs_record)
        
        for box_est, box_est_global in zip(boxes_est, boxes_est_global):
            box_est.score = box_est_global.detection_score

        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        plt.xlim(xmin=-30, xmax=30)
        plt.ylim(ymin=-30, ymax=30)

        result_dic = self.bevformer_results['map_results'][sample_token]['vectors']
        for vector in result_dic:
            if vector['confidence_level'] < 0.6:
                continue
            pred_pts_3d = vector['pts']
            pred_label_3d = vector['type']
            pts_x = np.array([pt[0] for pt in pred_pts_3d])
            pts_y = np.array([pt[1] for pt in pred_pts_3d])
            axes.plot(pts_x, pts_y, color=self.colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            axes.scatter(pts_x, pts_y, color=self.colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)  

        ignore_list = ['barrier', 'bicycle', 'traffic_cone']
        for i, box in enumerate(boxes_est):
            if box.name in ignore_list:
                continue
            if box.score < self.conf_th or abs(box.center[0]) > 15 or abs(box.center[1]) > 30:
                continue
            box.render(axes, view=np.eye(4), colors=('tomato', 'tomato', 'tomato'), linewidth=1, box_idx=None)
            if traj_use_perstep_offset:
                mode_idx = [0, 1, 2, 3, 4, 5]
                box.render_fut_trajs_grad_color(axes, linewidth=1, mode_idx=mode_idx, fut_ts=6, cmap='autumn')
            else:
                box.render_fut_trajs_coords(axes, color='tomato', linewidth=1)

        axes.plot([-0.9, -0.9], [-2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
        axes.plot([-0.9, 0.9], [2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
        axes.plot([0.9, 0.9], [2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
        axes.plot([0.9, -0.9], [-2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
        axes.plot([0.0, 0.0], [0.0, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
        
        plan_cmd = np.argmax(self.bevformer_results['plan_results'][sample_token][1][0,0,0])
        plan_traj = self.bevformer_results['plan_results'][sample_token][0][plan_cmd]
        plan_traj[abs(plan_traj) < 0.01] = 0.0
        plan_traj = plan_traj.cumsum(axis=0)
        plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
        plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)

        plan_vecs = None
        for i in range(plan_traj.shape[0]):
            plan_vec_i = plan_traj[i]
            x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
            y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
            xy = np.stack((x_linspace, y_linspace), axis=1)
            xy = np.stack((xy[:-1], xy[1:]), axis=1)
            if plan_vecs is None:
                plan_vecs = xy
            else:
                plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

        cmap = 'winter'
        y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
        colors = color_map(y[:-1], cmap)
        line_segments = LineCollection(plan_vecs, colors=colors, linewidths=1, linestyles='solid', cmap=cmap)
        axes.add_collection(line_segments)

        axes.axes.xaxis.set_ticks([])
        axes.axes.yaxis.set_ticks([])
        axes.axis('off')
        fig.set_tight_layout(True)
        fig.canvas.draw()
        plt.savefig(osp.join(self.save_path,'bev_pred.png'), bbox_inches='tight', dpi=200)
        plt.close()

    def obtain_sensor2top(self, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type='lidar'):
        sd_rec = self.nusc.get('sample_data', sensor_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        data_path = str(self.nusc.get_sample_data_path(sd_rec['token']))
        if os.getcwd() in data_path:
            data_path = data_path.split(f'{os.getcwd()}/')[-1]
        sweep = {
            'data_path': data_path,'type': sensor_type,'sample_data_token': sd_rec['token'],
            'sensor2ego_translation': cs_record['translation'],'sensor2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],'ego2global_rotation': pose_record['rotation'],
            'timestamp': sd_rec['timestamp']
        }

        l2e_r_s = sweep['sensor2ego_rotation']
        l2e_t_s = sweep['sensor2ego_translation']
        e2g_r_s = sweep['ego2global_rotation']
        e2g_t_s = sweep['ego2global_translation']

        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
        sensor2lidar_rotation = R.T
        sensor2lidar_translation = T
        return sensor2lidar_rotation, sensor2lidar_translation

    def render_sample_data(self, sample_token: str, traj_use_perstep_offset: bool = True):
        self.lidiar_render(sample_token, traj_use_perstep_offset=traj_use_perstep_offset)

    def run_visualization(self):
        """
        核心对外调用方法：执行完整可视化流程，生成视频
        """
        try:
            for id in tqdm(range(len(self.sample_token_list))):
                mmcv.mkdir_or_exist(self.save_path)
                # 生成BEV预测图
                self.render_sample_data(self.sample_token_list[id])
                pred_path = osp.join(self.save_path, 'bev_pred.png')
                pred_img = cv2.imread(pred_path)
                os.remove(pred_path)

                sample_token = self.sample_token_list[id]
                sample = self.nusc.get('sample', sample_token)
                cam_imgs = []
                # 生成6相机画面
                for cam in self.cams:
                    sample_data_token = sample['data'][cam]
                    sd_record = self.nusc.get('sample_data', sample_data_token)
                    sensor_modality = sd_record['sensor_modality']
                    if sensor_modality == 'camera':
                        boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                                    name=record['detection_name'], token='predicted') for record in self.bevformer_results['results'][sample_token]]
                        data_path, boxes_pred, camera_intrinsic = self.get_predicted_data(sample_data_token, pred_anns=boxes)
                        _, boxes_gt, _ = self.nusc.get_sample_data(sample_data_token, box_vis_level=BoxVisibility.ANY)
                        data = Image.open(data_path)

                        _, ax = plt.subplots(1, 1, figsize=(6, 12))
                        ax.imshow(data)
                        if cam == 'CAM_FRONT':
                            lidar_sd_record =  self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                            lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
                            lidar_pose_record = self.nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
                            plan_cmd = np.argmax(self.bevformer_results['plan_results'][sample_token][1][0,0,0])
                            plan_traj = self.bevformer_results['plan_results'][sample_token][0][plan_cmd]
                            plan_traj[abs(plan_traj) < 0.01] = 0.0
                            plan_traj = plan_traj.cumsum(axis=0)
                            plan_traj = np.concatenate((plan_traj[:, [0]],plan_traj[:, [1]],-1.0*np.ones((plan_traj.shape[0], 1)),np.ones((plan_traj.shape[0], 1))), axis=1)
                            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
                            plan_traj[0, 0] = 0.3
                            plan_traj[0, 2] = -1.0
                            plan_traj[0, 3] = 1.0

                            l2e_r = lidar_cs_record['rotation']
                            l2e_t = lidar_cs_record['translation']
                            e2g_r = lidar_pose_record['rotation']
                            e2g_t = lidar_pose_record['translation']
                            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                            e2g_r_mat = Quaternion(e2g_r).rotation_matrix
                            s2l_r, s2l_t = self.obtain_sensor2top(sample_data_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
                            lidar2cam_r = np.linalg.inv(s2l_r)
                            lidar2cam_t = s2l_t @ lidar2cam_r.T
                            lidar2cam_rt = np.eye(4)
                            lidar2cam_rt[:3, :3] = lidar2cam_r.T
                            lidar2cam_rt[3, :3] = -lidar2cam_t
                            viewpad = np.eye(4)
                            viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic
                            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                            plan_traj = lidar2img_rt @ plan_traj.T
                            plan_traj = plan_traj[0:2, ...] / np.maximum(plan_traj[2:3, ...], np.ones_like(plan_traj[2:3, ...]) * 1e-5)
                            plan_traj = plan_traj.T
                            plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)

                            plan_vecs = None
                            for i in range(plan_traj.shape[0]):
                                plan_vec_i = plan_traj[i]
                                x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
                                y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
                                xy = np.stack((x_linspace, y_linspace), axis=1)
                                xy = np.stack((xy[:-1], xy[1:]), axis=1)
                                if plan_vecs is None:
                                    plan_vecs = xy
                                else:
                                    plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

                            cmap = 'winter'
                            y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
                            colors = color_map(y[:-1], cmap)
                            line_segments = LineCollection(plan_vecs, colors=colors, linewidths=2, linestyles='solid', cmap=cmap)
                            ax.add_collection(line_segments)

                        ax.set_xlim(0, data.size[0])
                        ax.set_ylim(data.size[1], 0)
                        ax.axis('off')
                        savepath = osp.join(self.save_path, f'{cam}_PRED')
                        plt.savefig(savepath, bbox_inches='tight', dpi=200, pad_inches=0.0)
                        plt.close()

                        cam_img = cv2.imread(data_path)
                        lw = 6
                        tf = max(lw - 3, 1)
                        w, h = cv2.getTextSize(cam, 0, fontScale=lw / 6, thickness=tf)[0]
                        txt_color=(255, 255, 255)
                        cv2.putText(cam_img, cam, (10, h + 10),0,lw /6,txt_color,thickness=tf,lineType=cv2.LINE_AA)
                        cam_imgs.append(cam_img)

                # 拼接BEV图+添加文字
                plan_cmd = np.argmax(self.bevformer_results['plan_results'][sample_token][1][0,0,0])
                plan_cmd_str = self.cmd_list[plan_cmd]
                pred_img = cv2.copyMakeBorder(pred_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
                pred_img = cv2.putText(pred_img, 'BEV', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
                pred_img = cv2.putText(pred_img, plan_cmd_str, (20, 770), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3,cv2.LINE_AA)

                # 拼接相机画面+最终合成视频帧
                cam_img_top = cv2.hconcat([cam_imgs[0], cam_imgs[1], cam_imgs[2]])
                cam_img_down = cv2.hconcat([cam_imgs[3], cam_imgs[4], cam_imgs[5]])
                cam_img = cv2.vconcat([cam_img_top, cam_img_down])
                cam_img = cv2.resize(cam_img, (2133, 800))
                vis_img = cv2.hconcat([cam_img, pred_img])
                self.video.write(vis_img)
        finally:
            # 无论是否异常，都释放视频资源+销毁窗口
            self.video.release()
            cv2.destroyAllWindows()
            print(f"✅ 可视化完成！视频已保存至: {self.video_path}")