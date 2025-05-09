# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import json
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import torch


def is_point_inside_box(point, box):
    """Check if a point is inside the bounding box.

    Args:
        point (tuple): Tuple representing (x, y) coordinates of the point.
        box (torch.Tensor): Tensor representing (x_min, y_min, x_max, y_max) for the bounding box.

    Returns:
        bool: True if the point is inside the box, False otherwise.
    """
    x, y = point
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max


def calculate_center(box):
    """Calculate the center (x, y) of the bounding box.

    Args:
        box (torch.Tensor): Tensor representing (x_min, y_min, x_max, y_max) for the bounding box.

    Returns:
        tuple: Tuple representing (x_center, y_center) of the box.
    """
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center


def find_building_for_boxes(label_boxes, building_boxes):
    """Find the building for each label box based on their center points.

    Args:
        label_boxes (list): List of tensors, representing label boxes.
        building_boxes (list): List of tensors, representing building boxes.

    Returns:
        dict: Dictionary containing building indices as keys and a list of corresponding label boxes as values.
    """
    result_dict = {}

    for label_box in label_boxes:
        label_center = calculate_center(label_box[:4])
        building_found = False

        for j, building_box in enumerate(building_boxes):
            if is_point_inside_box(label_center, building_box[:4]):
                result_dict.setdefault(j, []).append(label_box.tolist())
                building_found = True
                break

        if not building_found:
            result_dict.setdefault('outside_building', []).append(label_box.tolist())

    return result_dict


@smart_inference_mode()
def run(
        weights='/root/autodl-tmp/autodl-tmp/yolov5/yolov5/runs_zihui/train/exp_5s22/weights/best.pt',  # model path or triton URL
        source='/root/autodl-tmp/autodl-tmp/yolov5/yolov5/city_data_component_zihui/Detecting Images_final/Chinatown 5rd round merge202505',  # file/dir/URL/glob/screen/0(webcam)
        data='/root/autodl-tmp/autodl-tmp/yolov5/yolov5/data/zihui_city_data_component_R2.yaml',  # dataset.yaml path
        imgsz=(1280, 1280),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        save_city_feature_path=None,
):
    # 程序会在开始处理图片之前自动创建保存结果的目录
    if save_city_feature_path:
        os.makedirs(save_city_feature_path, exist_ok=True)

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
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
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # 初始化一个空列表，用于存储feature vector矩阵
    matrices_list = []
    building_mertices_list = []
    metrices_embedding_list = []
    building_dic = {}
    building_dict_value_index = 0
    
    # 定义需要处理的标签列表
    '''
    names:
      0: Main Pilaster
      1: Secondary Pilaster  
      2: Long Window
      3: Casement Window
      4: Fanlight
      5: Modillion
      6: Festoon
      7: Chinese Plaque
      8: Green Glazed Tiles Canopy
      9: Chinese Decorative Panel
      10: Malay Transom
      11: Fretwork Fascia
      12: Majolica Tiles
      13: Stepping Parapet
      14: Modern Window
      15: Shades
      16: Building
    '''

    # 正确定义标签列表，确保与模型输出匹配
    labels_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # Building类的索引
    BUILDING_CLASS_INDEX = 16  # 根据注释，Building是类别16
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            if len(pred[0]) > 0:  # 确保检测结果不为空
                index_tensor = torch.arange(pred[0].shape[0]).float().unsqueeze(1).to(pred[0].device)
                pred[0] = torch.cat([pred[0], index_tensor], dim=1)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 分离建筑物和组件检测结果
                building_list_per_image = [det_i for det_i in det if det_i[5] == BUILDING_CLASS_INDEX]
                label_component_tensor = [det_i for det_i in det if det_i[5] != BUILDING_CLASS_INDEX]
                
                # 只有当有建筑物和组件时才进行处理
                if building_list_per_image and label_component_tensor:
                    building_results = find_building_for_boxes(label_component_tensor, building_list_per_image)

                    for building_result_i in building_results:
                        if building_result_i == 'outside_building':
                            continue
                            
                        # 记录建筑物信息
                        building_dic[str(building_dict_value_index)] = str(path)
                        building_dict_value_index += 1
                        
                        # 将建筑物边界框添加到列表
                        building_box = building_list_per_image[building_result_i][:4].cpu().numpy()
                        building_mertices_list.append(building_box)
                        
                        # 处理该建筑物中的组件
                        building_result = np.array(building_results[building_result_i])
                        building_result = torch.from_numpy(building_result)
                        
                        # 初始化特征向量（使用与标签数量匹配的大小）
                        per_building_class = np.zeros((1, len(labels_of_interest)), dtype=float)
                        
                        # 统计每个类别的组件数量
                        for c in building_result[:, 5].unique():
                            n = (building_result[:, 5] == c).sum()  # detections per class
                            c_int = int(c.item())
                            s += f"{n} {names[c_int]}{'s' * (n > 1)}, "  # add to string
                            
                            # 找到类别在感兴趣标签中的索引位置
                            if c_int in labels_of_interest:
                                idx = labels_of_interest.index(c_int)
                                per_building_class[0][idx] = float(n.item())
                                
                        matrices_list.append(per_building_class)
                
                # Write results (画出检测框)
                for *xyxy, conf, cls in reversed(det[:, :6]):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # 不显示Building类别的边界框
                        if c == BUILDING_CLASS_INDEX:
                            continue
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    
    # 检查是否有数据可用于保存
    if not matrices_list or not building_mertices_list:
        LOGGER.warning("No data collected to save. Check if detections are occurring.")
        return None, None
    
    # 保存结果
    if save_city_feature_path:
        # 拼接结果矩阵
        concatenated_matrix = np.concatenate(matrices_list, axis=0)
        concatenated_building_matrix = np.array([arr for arr in building_mertices_list])
        
        # 设置保存路径
        save_file_path = os.path.join(save_city_feature_path, 'component_feature_vectors.npy')
        save_file_buildingh_path = os.path.join(save_city_feature_path, 'building_boxes.npy')
        save_json_path = os.path.join(save_city_feature_path, 'building_image_mapping.json')
        
        # 保存文件
        np.save(save_file_path, concatenated_matrix)
        np.save(save_file_buildingh_path, concatenated_building_matrix)
        
        # 将字典保存为JSON文件
        with open(save_json_path, 'w') as json_file:
            json.dump(building_dic, json_file, indent=4)
        
        LOGGER.info(f"Saved feature vectors to {save_file_path}")
        LOGGER.info(f"Saved building boxes to {save_file_buildingh_path}")
        LOGGER.info(f"Saved building-image mapping to {save_json_path}")

    # Print results
    if len(matrices_list) == 1:
        return matrices_list, im0
    
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/root/autodl-tmp/autodl-tmp/yolov5/yolov5/runs_zihui/train/exp_5s22/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='/root/autodl-tmp/autodl-tmp/yolov5/yolov5/city_data_component_zihui/Detecting Images_final/Chinatown 5rd round merge202505', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='/root/autodl-tmp/autodl-tmp/yolov5/yolov5/data/zihui_city_data_component_R2.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', default=True, help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--save_city_feature_path', type=str, default='/root/autodl-tmp/yolov5/yolov5', help='save path for feature vectors')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)