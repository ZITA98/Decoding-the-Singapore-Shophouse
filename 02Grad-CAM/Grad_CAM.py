import sys
# 添加模块路径
sys.path.append("e:/13 Singapore shophouse/explaination_analysis_for_model/explaination_analysis_for_model")
sys.path.append("e:/13 Singapore shophouse/explaination_analysis_for_model/classification_model")

import torch.nn as nn
from model import Resnet_50  # type: ignore
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
import PIL
from scipy.ndimage import gaussian_filter
import os

# 设置路径和参数
model_path = "E:/13 Singapore shophouse/explaination_analysis_for_model/classification_model/origin_model.pth"
input_folder = r"E:\13 Singapore shophouse\explaination_analysis_for_model\explaination_analysis_for_model_0412\gsv_ld"
output_dir = r"E:\13 Singapore shophouse\explaination_analysis_for_model\explaination_analysis_for_model_0412\ld-gsv-4-2"
os.makedirs(output_dir, exist_ok=True)

# 目标类别 - 可修改为[1,0,0,0]、[0,1,0,0]、[0,0,1,0]或[0,0,0,1]
target_class = torch.tensor([[0, 0, 0, 1]])  

# 加载和准备模型
resnet_model = nn.DataParallel(Resnet_50(num_classes=4))
resnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
resnet_model.eval()

# 获取文件夹中的所有图片
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
image_files = []
for filename in os.listdir(input_folder):
    if os.path.splitext(filename.lower())[1] in image_extensions:
        image_files.append(os.path.join(input_folder, filename))

print(f"找到 {len(image_files)} 个图片文件")

# 处理每个图片
for img_path in image_files:
    # 提取文件名（不包括扩展名）
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    print(f"处理图片: {base_name}")
    
    # 定义全局变量和钩子函数
    gradients, activations = None, None

    def backward_hook(module, grad_input, grad_output):
        global gradients
        gradients = grad_output

    def forward_hook(module, input, output):
        global activations
        activations = output

    # 注册钩子
    backward_handle = resnet_model.module.resnet50.layer4[-1].register_full_backward_hook(hook=backward_hook)
    forward_handle = resnet_model.module.resnet50.layer4[-1].register_forward_hook(hook=forward_hook)

    # 加载和预处理图像
    image = Image.open(img_path).convert('RGB')
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform_test(image).to(resnet_model.src_device_obj)

    # 前向传播和反向传播
    output = resnet_model(img_tensor.unsqueeze(0))
    resnet_model(img_tensor.unsqueeze(0)).backward(target_class.to(resnet_model.src_device_obj))

    # 计算Grad-CAM热力图
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = F.relu(torch.mean(activations, dim=1).squeeze())
    heatmap /= torch.max(heatmap)

    # 保存原始热力图
    plt.figure(figsize=(10, 8), dpi=300)
    plt.matshow(heatmap.cpu().detach(), fignum=1)
    plt.title("Raw Class Activation Map")
    plt.colorbar(label="Activation Intensity")
    plt.savefig(os.path.join(output_dir, f"{base_name}_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 创建叠加可视化
    plt.figure(figsize=(12, 10), dpi=300)
    ax = plt.subplot(111)
    ax.axis('off')

    # 显示原始图像
    denormalized_img = img_tensor.clone().detach().cpu()
    denormalized_img[0] = denormalized_img[0] * 0.229 + 0.485
    denormalized_img[1] = denormalized_img[1] * 0.224 + 0.456
    denormalized_img[2] = denormalized_img[2] * 0.225 + 0.406
    denormalized_img = torch.clamp(denormalized_img, 0, 1)
    ax.imshow(to_pil_image(denormalized_img, mode='RGB'))

    # 优化热力图处理流程
    # 上采样并平滑热力图
    heatmap_upsampled = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0), 
        size=(256, 256), 
        mode='bilinear', 
        align_corners=True
    ).squeeze().cpu().detach().numpy()

    # 应用高斯平滑
    heatmap_smoothed = gaussian_filter(heatmap_upsampled, sigma=2)

    # 直接应用颜色映射
    colored_heatmap = (255 * cm.jet(heatmap_smoothed ** 1.5)[:, :, :3]).astype(np.uint8)

    # 叠加显示
    ax.imshow(colored_heatmap, alpha=0.35, interpolation='bilinear')
    plt.tight_layout()
    plt.title("Grad-CAM Visualization")
    plt.savefig(os.path.join(output_dir, f"{base_name}_overlay.png"), dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图表而不是显示，以避免显示太多窗口

    # 清理钩子
    backward_handle.remove()
    forward_handle.remove()
    
    print(f"已保存: {base_name}_heatmap.png 和 {base_name}_overlay.png")

print(f"Grad-CAM 可视化完成。所有结果保存在: {output_dir}")