# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection with CLASS-SPECIFIC confidence thresholds
Modified version: Different thresholds for different component classes
"""

import sys
import os
from pathlib import Path

# 设置YOLOv5根目录的绝对路径
YOLOV5_PATH = Path("/Users/tzh/Projects/AI/models/yolov5").resolve()
PROJECT_ROOT = Path("/Users/tzh/Library/CloudStorage/OneDrive-个人/01_PROJECT/05-shophouse/Shophouse_STYLE/05-CODE/Components_Detection").resolve()

# 添加到Python路径
if str(YOLOV5_PATH) not in sys.path:
    sys.path.append(str(YOLOV5_PATH))

ROOT = PROJECT_ROOT

import argparse
import platform
import numpy as np
import json
import torch
import datetime

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import torch


# ============================================================================
# 🎯 类别特定置信度阈值配置
# ============================================================================
CLASS_SPECIFIC_CONF_THRES = {
    # 高性能装饰类别 - 降低阈值，宁可多检不可漏检
    11: 0.40,  # Fretwork Fascia (AP=0.995) ⭐ 装饰细节，降低阈值
    12: 0.40,  # Majolica Tiles (AP=0.995) ⭐ 装饰细节，降低阈值
    
    # 高性能但不常见的类别 - 使用较高阈值
    7: 0.60,   # Chinese Plaque (AP=0.995) - 特征明显，保持较高阈值
    9: 0.60,   # Chinese Decorative Panel (AP=0.995) - 特征明显
    10: 0.60,  # Malay Transom (AP=0.995) - 特征明显
    
    # 结构性重要组件 - 降低阈值确保不漏检
    0: 0.40,   # Main Pilaster (AP=0.941) ⭐ 主柱很重要，降低阈值
    1: 0.45,   # Secondary Pilaster (AP=0.937) - 次要柱子
    
    # 窗户类别 - 使用中等阈值
    2: 0.45,   # Long Window (AP=0.979)
    3: 0.45,   # Casement Window (AP=0.927) ⭐ 优先级高于Modern Window
    4: 0.45,   # Fanlight (AP=0.954)
    14: 0.45,  # Modern Window (AP=0.802) ⚠️ 低优先级，会被Casement覆盖
    
    # 装饰组件 - 降低阈值，重视召回率
    5: 0.40,   # Modillion (AP=0.863) ⭐ 装饰组件，降低阈值
    6: 0.45,   # Festoon (AP=0.907)
    8: 0.45,   # Green Glazed Tiles Canopy (AP=0.924)
    15: 0.45,  # Shades (AP=0.907)
    
    # 最难检测的类别 - 使用最低阈值
    13: 0.35,  # Stepping Parapet (AP=0.776) ⚠️ 最低阈值
    
    # Building类别（将被跳过）
    16: 0.50,  # Building (不使用)
}

# 默认阈值（用于未指定的类别）
DEFAULT_CONF_THRES = 0.45

# ============================================================================
# 🔧 类别冲突解决配置
# ============================================================================
# 定义哪些类别对会发生冲突，以及优先保留哪个
# 格式: (低优先级类别, 高优先级类别, IoU阈值)
CLASS_CONFLICT_RULES = [
    (14, 3, 0.5),   # Modern Window vs Casement Window: 当IoU>0.5时，保留Casement Window
    # 可以添加更多冲突规则，例如:
    # (5, 0, 0.6),  # Modillion vs Main Pilaster: 当IoU>0.6时，保留Main Pilaster
]

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)
    
    Args:
        box1, box2: [x1, y1, x2, y2] 格式的边界框
    
    Returns:
        IoU值 (0-1之间)
    """
    # 计算交集区域
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 如果没有交集
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    # 交集面积
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 各自的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 并集面积
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def resolve_class_conflicts(detections, conflict_rules):
    """
    解决类别冲突：当两个不同类别的检测框重叠度高时，根据优先级保留其中一个
    
    Args:
        detections: 检测结果 tensor (n, 6+) [x1, y1, x2, y2, conf, cls, ...]
        conflict_rules: 冲突规则列表 [(低优先级类别, 高优先级类别, IoU阈值), ...]
    
    Returns:
        tuple: (解决冲突后的检测结果, 移除统计字典)
    """
    # 初始化移除统计
    removed_count = {rule[0]: 0 for rule in conflict_rules}
    
    if len(detections) == 0:
        return detections, removed_count
    
    # 转换为可操作的格式
    keep_mask = torch.ones(len(detections), dtype=torch.bool, device=detections.device)
    
    # 遍历每个冲突规则
    for low_priority_cls, high_priority_cls, iou_threshold in conflict_rules:
        # 找出低优先级和高优先级类别的检测
        low_priority_indices = [i for i, det in enumerate(detections) 
                               if int(det[5].item()) == low_priority_cls and keep_mask[i]]
        high_priority_indices = [i for i, det in enumerate(detections) 
                                if int(det[5].item()) == high_priority_cls and keep_mask[i]]
        
        # 检查每对可能冲突的检测
        for low_idx in low_priority_indices:
            low_box = detections[low_idx][:4].cpu().numpy()
            
            for high_idx in high_priority_indices:
                high_box = detections[high_idx][:4].cpu().numpy()
                
                # 计算IoU
                iou = calculate_iou(low_box, high_box)
                
                # 如果IoU超过阈值，移除低优先级的检测
                if iou > iou_threshold:
                    keep_mask[low_idx] = False
                    removed_count[low_priority_cls] += 1
                    break  # 已经决定移除，不需要继续检查
    
    # 返回过滤后的结果和统计信息
    filtered_detections = detections[keep_mask]
    
    return filtered_detections, removed_count
# ============================================================================


def apply_class_specific_thresholds(detections, class_thresholds, default_threshold):
    """
    应用类别特定的置信度阈值过滤检测结果
    
    Args:
        detections: 检测结果 tensor (n, 6+) [x1, y1, x2, y2, conf, cls, ...]
        class_thresholds: 类别到阈值的映射字典
        default_threshold: 默认阈值
    
    Returns:
        过滤后的检测结果
    """
    if len(detections) == 0:
        return detections
    
    # 创建掩码，标记哪些检测应该保留
    keep_mask = torch.zeros(len(detections), dtype=torch.bool, device=detections.device)
    
    for i, det in enumerate(detections):
        conf = det[4].item()
        cls = int(det[5].item())
        
        # 获取该类别的阈值
        threshold = class_thresholds.get(cls, default_threshold)
        
        # 如果置信度超过阈值，保留该检测
        if conf >= threshold:
            keep_mask[i] = True
    
    # 返回过滤后的结果
    filtered_detections = detections[keep_mask]
    
    return filtered_detections


def print_threshold_config(class_names):
    """
    打印类别特定阈值配置
    """
    LOGGER.info("="*80)
    LOGGER.info("📊 类别特定置信度阈值配置")
    LOGGER.info("="*80)
    
    # 按阈值分组
    threshold_groups = {}
    for cls_id, threshold in CLASS_SPECIFIC_CONF_THRES.items():
        if cls_id == 16:  # 跳过Building
            continue
        if threshold not in threshold_groups:
            threshold_groups[threshold] = []
        threshold_groups[threshold].append((cls_id, class_names[cls_id]))
    
    # 按阈值从高到低排序
    for threshold in sorted(threshold_groups.keys(), reverse=True):
        LOGGER.info(f"\n阈值 = {threshold:.2f}:")
        for cls_id, cls_name in threshold_groups[threshold]:
            ap_info = {
                7: "0.995", 9: "0.995", 10: "0.995", 11: "0.995", 12: "0.995",
                0: "0.941", 1: "0.937", 2: "0.979", 4: "0.954", 6: "0.907",
                8: "0.924", 15: "0.907", 3: "0.927", 5: "0.863",
                13: "0.776", 14: "0.802"
            }
            ap = ap_info.get(cls_id, "N/A")
            
            # 标记特殊配置的类别
            if cls_id in [0, 5, 11, 12]:
                priority_mark = " ⭐ 降低阈值（重要/装饰组件）"
            elif cls_id == 3:
                priority_mark = " ⭐ 高优先级"
            elif cls_id == 14:
                priority_mark = " ⚠️ 低优先级"
            elif cls_id == 13:
                priority_mark = " ⚠️ 最难检测"
            else:
                priority_mark = ""
            
            LOGGER.info(f"  • {cls_name:30s} (类别{cls_id:2d}, AP={ap}){priority_mark}")
    
    LOGGER.info(f"\n默认阈值（未指定类别）: {DEFAULT_CONF_THRES:.2f}")
    
    # 打印冲突解决规则
    LOGGER.info("\n" + "-"*80)
    LOGGER.info("🔧 类别冲突解决规则")
    LOGGER.info("-"*80)
    for low_cls, high_cls, iou_thresh in CLASS_CONFLICT_RULES:
        LOGGER.info(f"  • {class_names[low_cls]} vs {class_names[high_cls]}: "
                   f"当IoU>{iou_thresh}时，优先保留 {class_names[high_cls]}")
    LOGGER.info("="*80 + "\n")


@smart_inference_mode()
def run(
        weights='/Users/tzh/Library/CloudStorage/OneDrive-个人/01_PROJECT/05-shophouse/Shophouse_STYLE/05-CODE/Components_Detection/runs_zihui/train/exp_5s22/weights/best.pt',
        source='/Users/tzh/Library/CloudStorage/OneDrive-个人/01_PROJECT/05-shophouse/Shophouse_STYLE/Image Data/Crop20251003_Chinatown 6rd round merge202505yolopadding150',
        data='/Users/tzh/Library/CloudStorage/OneDrive-个人/01_PROJECT/05-shophouse/Shophouse_STYLE/05-CODE/Decoding-the-Singapore-Shophouse/03Yolov5/data_component.yaml',
        imgsz=(1280, 1280),
        conf_thres=0.25,  # 这个值作为NMS的初始阈值，实际过滤使用类别特定阈值
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project='/Users/tzh/Library/CloudStorage/OneDrive-个人/01_PROJECT/05-shophouse/Shophouse_STYLE/05-CODE/Components_Detection/runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
        save_city_feature_path=None,
):
    # 打印开始信息
    LOGGER.info(f"Script started with CLASS-SPECIFIC thresholds")
    LOGGER.info(f"Source: {source}")
    LOGGER.info(f"Save city feature path: {save_city_feature_path}")
    
    # 使用时间戳创建保存目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_city_feature_path:
        try:
            save_dir_with_timestamp = os.path.join(save_city_feature_path, timestamp)
            os.makedirs(save_dir_with_timestamp, exist_ok=True)
            LOGGER.info(f"Created timestamp directory: {save_dir_with_timestamp}")
        except Exception as e:
            LOGGER.error(f"Failed to create timestamp directory: {e}")
            save_dir_with_timestamp = None
    else:
        save_dir_with_timestamp = None

    if save_city_feature_path:
        os.makedirs(save_city_feature_path, exist_ok=True)

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    
    # 打印类别特定阈值配置
    print_threshold_config(names)

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    # 用于统计
    image_component_stats = []
    labels_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    BUILDING_CLASS_INDEX = 16
    
    # 统计每个类别被过滤的数量
    class_filtered_stats = {i: {'total': 0, 'filtered': 0} for i in range(17)}
    conflict_removed_total = {rule[0]: 0 for rule in CLASS_CONFLICT_RULES}
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS (使用较低的初始阈值，让更多检测通过，后面用类别特定阈值过滤)
        with dt[2]:
            # 根据YOLOv5版本使用不同的参数
            try:
                # 尝试新版本的参数
                pred = non_max_suppression(pred, conf_thres=0.20, iou_thres=iou_thres, 
                                           classes=classes, agnostic=agnostic_nms, max_det=max_det)
            except TypeError:
                try:
                    # 尝试旧版本的参数（没有agnostic参数）
                    pred = non_max_suppression(pred, conf_thres=0.20, iou_thres=iou_thres, 
                                               classes=classes, max_det=max_det)
                except TypeError:
                    # 最基础的版本
                    pred = non_max_suppression(pred, conf_thres=0.20, iou_thres=iou_thres)

        # Process predictions
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # 统计NMS后的检测数量
                det_before_filter = det.clone()
                for d in det_before_filter:
                    cls = int(d[5].item())
                    class_filtered_stats[cls]['total'] += 1
                
                # 🎯 应用类别特定阈值过滤
                det = apply_class_specific_thresholds(det, CLASS_SPECIFIC_CONF_THRES, DEFAULT_CONF_THRES)
                
                # 🔧 解决类别冲突（例如: Modern Window vs Casement Window）
                det, conflict_removed = resolve_class_conflicts(det, CLASS_CONFLICT_RULES)
                
                # 累计冲突解决统计
                for cls_id, count in conflict_removed.items():
                    conflict_removed_total[cls_id] += count
                
                # 统计过滤后的检测数量
                for d in det:
                    cls = int(d[5].item())
                    class_filtered_stats[cls]['filtered'] += 1
                
                # 只处理非Building的组件
                component_detections = [det_i for det_i in det if det_i[5] != BUILDING_CLASS_INDEX]
                
                LOGGER.info(f"Image {seen}/{len(dataset)}: {p.name}: "
                           f"Found {len(component_detections)} components after class-specific filtering")

                # 初始化特征向量
                per_image_class_count = np.zeros((1, len(labels_of_interest)), dtype=float)
                
                # 统计每个类别
                if component_detections:
                    component_tensor = torch.stack(component_detections)
                    
                    for c in component_tensor[:, 5].unique():
                        n = (component_tensor[:, 5] == c).sum()
                        c_int = int(c.item())
                        s += f"{n} {names[c_int]}{'s' * (n > 1)}, "
                        
                        if c_int in labels_of_interest:
                            idx = labels_of_interest.index(c_int)
                            per_image_class_count[0][idx] = float(n.item())
                
                image_component_stats.append({
                    'image_path': str(path),
                    'image_name': p.name,
                    'feature_vector': per_image_class_count,
                    'component_count': len(component_detections)
                })
                
                # Write results
                for j, (*xyxy, conf, cls) in enumerate(det.tolist()):
                    c = int(cls)
                    
                    if c == BUILDING_CLASS_INDEX:
                        continue
                    
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    if save_img or save_crop or view_img:
                        # 显示该检测使用的阈值
                        threshold_used = CLASS_SPECIFIC_CONF_THRES.get(c, DEFAULT_CONF_THRES)
                        label = None if hide_labels else (
                            f'{names[c]} {conf:.2f}' if hide_conf else 
                            f'{names[c]} {conf:.2f} (T:{threshold_used:.2f})'
                        )
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if save_crop:
                        crop_filename = f'{p.stem}_{names[c]}_{j:02d}.jpg'
                        save_one_box(xyxy, imc, 
                                    file=save_dir / 'crops' / names[c] / crop_filename, 
                                    BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # Save results
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    
    # 打印过滤统计
    LOGGER.info("\n" + "="*80)
    LOGGER.info("📊 类别特定阈值过滤统计")
    LOGGER.info("="*80)
    for cls_id in sorted(class_filtered_stats.keys()):
        if cls_id == 16:  # 跳过Building
            continue
        stats = class_filtered_stats[cls_id]
        if stats['total'] > 0:
            threshold = CLASS_SPECIFIC_CONF_THRES.get(cls_id, DEFAULT_CONF_THRES)
            kept_pct = (stats['filtered'] / stats['total']) * 100
            LOGGER.info(f"{names[cls_id]:30s} (T={threshold:.2f}): "
                       f"{stats['filtered']:4d} / {stats['total']:4d} kept ({kept_pct:5.1f}%)")
    
    # 打印冲突解决统计
    if any(conflict_removed_total.values()):
        LOGGER.info("\n" + "-"*80)
        LOGGER.info("🔧 类别冲突解决统计")
        LOGGER.info("-"*80)
        for low_cls, high_cls, iou_thresh in CLASS_CONFLICT_RULES:
            removed = conflict_removed_total.get(low_cls, 0)
            if removed > 0:
                LOGGER.info(f"{names[low_cls]:30s} 因与 {names[high_cls]} 冲突被移除: {removed} 次")
    
    LOGGER.info("="*80 + "\n")
    
    # 保存结果
    if save_city_feature_path and save_dir_with_timestamp and image_component_stats:
        try:
            all_feature_vectors = np.concatenate([stat['feature_vector'] for stat in image_component_stats], axis=0)
            
            image_mapping = {
                str(idx): {
                    'image_path': stat['image_path'],
                    'image_name': stat['image_name'],
                    'component_count': stat['component_count']
                }
                for idx, stat in enumerate(image_component_stats)
            }
            
            save_file_path = os.path.join(save_dir_with_timestamp, f'{timestamp}_component_feature_vectors.npy')
            save_json_path = os.path.join(save_dir_with_timestamp, f'{timestamp}_image_mapping.json')
            save_summary_path = os.path.join(save_dir_with_timestamp, f'{timestamp}_detection_summary.txt')
            
            np.save(save_file_path, all_feature_vectors)
            with open(save_json_path, 'w') as json_file:
                json.dump(image_mapping, json_file, indent=4)
            
            with open(save_summary_path, 'w') as f:
                f.write(f"Detection Summary (Class-Specific Thresholds)\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Total images processed: {len(image_component_stats)}\n")
                f.write(f"Total components detected: {sum(stat['component_count'] for stat in image_component_stats)}\n\n")
                f.write(f"Class-Specific Thresholds Used:\n")
                for cls_id, threshold in sorted(CLASS_SPECIFIC_CONF_THRES.items()):
                    if cls_id != 16:
                        f.write(f"  {names[cls_id]:30s}: {threshold:.2f}\n")
            
            LOGGER.info(f"Successfully saved results to {save_dir_with_timestamp}")
            
        except Exception as e:
            LOGGER.error(f"Error during saving: {e}")
    
    t = tuple(x.t / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/Users/tzh/Library/CloudStorage/OneDrive-个人/01_PROJECT/05-shophouse/Shophouse_STYLE/05-CODE/Components_Detection/runs_zihui/train/exp_5s22/weights/best.pt')
    parser.add_argument('--source', type=str, default='/Users/tzh/Library/CloudStorage/OneDrive-个人/01_PROJECT/05-shophouse/Shophouse_STYLE/Image Data/Detecting Images_final/Crop20251007_Chinatown 6rd round merge202505yolopadding150')
    parser.add_argument('--data', type=str, default='/Users/tzh/Library/CloudStorage/OneDrive-个人/01_PROJECT/05-shophouse/Shophouse_STYLE/05-CODE/Decoding-the-Singapore-Shophouse/03Yolov5/data_component.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280])
    parser.add_argument('--conf-thres', type=float, default=0.20, help='initial NMS threshold (will use class-specific after)')
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='')
    parser.add_argument('--view-img', action='store_true')
    parser.add_argument('--save-txt', action='store_true', default=True)
    parser.add_argument('--save-conf', action='store_true', default=True)
    parser.add_argument('--save-crop', action='store_true', default=True)
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--project', default=ROOT / 'runs_zihui/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-thickness', default=3, type=int)
    parser.add_argument('--hide-labels', default=False, action='store_true')
    parser.add_argument('--hide-conf', default=False, action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--vid-stride', type=int, default=1)
    parser.add_argument('--save_city_feature_path', type=str, default='/Users/tzh/Library/CloudStorage/OneDrive-个人/01_PROJECT/05-shophouse/Shophouse_STYLE/05-CODE/Components_Detection/zihui_components_detection')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)