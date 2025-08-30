# Decoding the Singapore Shophouse
## Overview

Through visual intelligence technologies, this repository implements a computational framework for systematically analyzing vernacular architecture --- Singapore Shophouse. The framework augments traditional intuitive analysis by applying machine learning to identify formal patterns, classification hierarchies, and evolutionary relationships in historical buildings.

This repository contains a building analysis system with four key components:

1. **ResNet-50 Façade-based Cultural Classification** - Initial classifier for cultural analysis across four historical districts using façade imagery and ImageNet pre-trained weights

2. **Grad-CAM Visualization** - Provides visual explanations for classification decisions

3. **YOLOv5 Building Component Detection** - Detects and localizes architectural components

4. **Phylogenetic Analysis** - Constructs evolutionary relationships using external software (PAST 4.17 & SplitsTree6 6.4.7) with provided workflows and interface scripts
   

## Repository Structure

```
building-analysis/
├── classification_model/      # Style classification model
│   ├── custom_dataset         # Custom dataset implementation
│   ├── Dataset_city           # City building dataset
│   ├── model                  # Model definition
│   ├── resnet_model_train     # Training script
│   ├── resnet_test            # Testing script
│   └── split_*_data           # Data splitting utilities
│
├── gradcam/                   # Explainable AI component
│   └── gradcam.py             # Grad-CAM implementation
│
├── yolov5/                    # Building component detection
│   ├── models/                # Model architectures
│   ├── utils/                 # Utility functions
│   ├── weights/               # Model weights
│   ├── train                  # Training script
│   ├── val                    # Validation script
│   ├── confusion_matrix       # Evaluation results
│   ├── detect_label_per_building  # Detection script
│   ├── export                 # Export script
│   └── data_component  # Dataset configuration
│
├── data/                      # Data directory
│   └── README.md              # Dataset information and download links

└── outputs/                   # Output directory for results
    ├── yolo_detections/       # YOLOv5 detection results
    └── gradcam_visualizations/  # Grad-CAM visualization results
    └── demo/                  # Demo analysis results (user-generated)
       ├── gradcam/          # Demo Grad-CAM outputs
       └── detection/         # Demo detection outputs
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

The main dependencies include:

* PyTorch (>=1.11.0)

* OpenCV (>=4.7.0)

* Ultralytics YOLOv5

* NumPy, Pandas, Matplotlib

* TensorBoard for visualization

### Required Non-Standard Hardware
- **NVIDIA GPU**: CUDA-compatible GPU required for training and efficient inference
- **Minimum GPU Memory**: 4GB VRAM (8GB recommended for training)
- **Note**: CPU-only execution possible but significantly slower
  

## Data
### Singapore Shophouse Dataset
- **Size**: shophouse images from Singapore four conservation district （640×640 pixels GSV /3024×4032 on-site image）
- **Annotations**: 708 annotated photographs with 17 label classes (16 architectural elements + building class)
- **Elements**: Main pilaster, fanlight, secondary pilaster, festoon, modillion, Chinese plaque, Chinese decorative panel, Green glazed canopy, Malay transom, fretwork fascia, majolica tiles, long window, modern window, tripart casement window, shades, and stepping parapet.

For detailed dataset information, please refer to Readme(Dataset)md by google link.

The dataset is available via Google Drive: [Singapore Shophouse Analysis Dataset]
The data supporting the findings of this study are currently embargoed during the peer review process.The confidential access link to the data has been shared with you separately via the editorial office. Upon formal acceptance of the manuscript, the data will be made publicly available on Zenodo.

You can download it manually and place it in the appropriate directories.


## Usage Instructions

### 1. Data Preparation

### 2. Download Pre-trained Weights

#### ResNet-50 Classification Weights
```bash
# Download our trained façade-based cultural classification model from Google Drive
# Large file size requires external hosting
# Download link: https://drive.google.com/open?id=1ZFB3eDTWy1JnOz9qBjo4b1rktaGksEVV&usp=drive_copy
# Place the downloaded weights in: classification_model/weights/best_model.pth
```

#### YOLOv5 Base Weights (included in repository)
```bash
# YOLOv5 base weights are already included in yolov5/weights/ directory
ls yolov5/weights/  # Should show yolov5s.pt and custom trained weights
```

### 3. Training Models (Optional - Pre-trained weights provided)

#### ResNet-50 Façade-based Classification
```bash
cd classification_model
# Train ResNet-50 classifier on façade imagery across four historical districts
# Parameters: 50 epochs, batch_size 32, lr 0.001, ImageNet pre-trained weights
python resnet_model_train.py --epochs 50 --batch_size 32 --lr 0.001 --split_ratio 0.8
```

#### Façade Element Detection
```bash
cd yolov5
# Train YOLOv5 for 16 façade elements detection
# Parameters: 120 epochs, batch_size 16, lr 0.01, cosine annealing, momentum 0.934
python train.py --data data_component.yaml --epochs 120 --batch_size 16 --lr 0.01 \
    --weights yolov5s.pt --cache --cos-lr --momentum 0.934 --weight-decay 0.0005
```

### 4. Running Analysis

#### Grad-CAM Visualizations
```bash
# Generate explainable AI visualizations for façade element contributions
python gradcam/gradcam.py --model_path classification_model/weights/best_model.pth --image_path data/sample_image.jpg
```

#### Façade-based Cultural Classification and Element Detection
```bash
# Run ResNet-50 cultural classification on façade imagery across four historical districts
python classification_model/resnet_test.py --model_path classification_model/weights/best_model.pth --test_dir data/test

# Detect 16 architectural elements with confidence threshold 0.4 and IoU 0.45
python yolov5/detect_label_per_building.py --weights yolov5/weights/best.pt \
    --source data/test_images --conf 0.4 --iou 0.45
```

### 5. Phylogenetic Network Analysis

**Note**: This analysis uses external open-source software. We provide workflows and interface scripts to reproduce the phylogenetic networks described in our paper.

#### Install Required Software
**PAST (PAleontological STatistics):**

Download PAST 4.17 from official sources:
[University of Oslo](https://www.nhm.uio.no/english/research/infrastructure/past/)

Follow the installation instructions for your operating system.

**SplitsTree6:**

Download SplitsTree6 6.4.7 from official sources:
- Primary: [University of Tübingen](https://uni-tuebingen.de/en/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/algorithms-in-bioinformatics/software/splitstree/)
- Download: [Software Portal](https://software-ab.cs.uni-tuebingen.de/download/splitstree/welcome.html)
- GitHub: [husonlab/splitstree6](https://github.com/husonlab/splitstree6)

Follow the installation guide provided on the website.

## Expected Runtime: 

- ResNet-50 training: 2-3 hours (50 epochs, batch_size 32, RTX 3080)
- YOLOv5 training: 8-12 hours (120 epochs, batch_size 16, RTX 3080)
- Inference: ~50ms per image for cultural classification, ~100ms for element detection
- Feature extraction and distance calculation: 15-30 minutes
- Phylogenetic analysis in PAST/SplitsTree6: 10-20 minutes (manual workflow)
- Post-processing and visualization: 5-10 minutes
- Complete phylogenetic network analysis: 45-75 minutes total


## Results

### 1. Classification Results
<img width="1953" height="911" alt="图片1" src="https://github.com/user-attachments/assets/a6c535cd-fe65-43e4-9706-4df64a8e4e53" />

### 2. Component Detection
The YOLOv5 component includes a confusion matrix visualization for evaluation of detection performance. This can be found in the yolov5/confusion_matrix directory.
<img width="6283" height="2995" alt="资源 9@400x" src="https://github.com/user-attachments/assets/f50336fd-d2b4-491c-9016-4bcee7d1a202" />

![confusion_matrix](https://github.com/user-attachments/assets/41bfb48c-3812-48f0-9f82-caa0d7f9e9be)

### 3. Phylogenetic Network
<img width="6301" height="5320" alt="资源 1@400x" src="https://github.com/user-attachments/assets/630b7187-b89a-46bf-b4a9-51e78bdb016e" />



## Quick Demo

Users can test the framework using the complete dataset following the Usage Instructions above. The expected outputs and runtime are detailed in the Results section.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

##

