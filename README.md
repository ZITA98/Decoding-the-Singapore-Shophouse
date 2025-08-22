# Decoding the Singapore Shophouse
## Overview

This repository contains a building analysis system with three key components:

1. **Architectural Style Classification** - Identifies building styles using deep learning

2. **Grad-CAM Visualization** - Provides visual explanations for classification decisions

3. **YOLOv5 Building Component Detection** - Detects and localizes architectural components

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

### Data

The dataset is available via Google Drive: [Singapore Shophouse Analysis Dataset]

You can download it manually and place it in the appropriate directories.

## Usage Instructions

### 1. Data Preparation
```bash
# Organize your dataset
python classification_model/split_train_data.py --data_dir data/images --split_ratio 0.8

# Prepare YOLO annotations
python yolov5/utils/convert_annotations.py --input data/annotations --output yolov5/datasets
```

### 2. Training Models

#### Style Classification
```bash
cd classification_model
python resnet_model_train.py --epochs 100 --batch_size 32 --lr 0.001
```

#### Component Detection
```bash
cd yolov5
python train.py --data data_component.yaml --epochs 300 --weights yolov5s.pt --cache
```

### 3. Running Analysis
```bash
# Classify shophouse styles
python classification_model/resnet_test.py --model_path models/best_model.pth --test_dir data/test

# Detect architectural components
python yolov5/detect_label_per_building.py --weights yolov5/weights/best.pt --source data/test_images

# Generate Grad-CAM visualizations
python gradcam/gradcam.py --model_path models/best_model.pth --image_path data/sample_image.jpg
```

### 4. Phylogenetic Network Analysis

**Generate morphological distance matrices:**
```bash
# Extract architectural features and compute distance matrices
python phylogenetic_analysis/compute_morphological_distances.py --features_dir outputs/features

# Export distance matrix for PAST analysis
python phylogenetic_analysis/export_for_past.py --distance_matrix outputs/distance_matrix.csv
```

**Run phylogenetic analysis in PAST:**
```bash
# Open PAST software and import distance matrix
# Follow the script: phylogenetic_analysis/past_scripts/neighbor_net_analysis.txt
# This generates the phylogenetic network and statistical tests
```

**Construct networks in SplitsTree6:**
```bash
# Import distance matrix into SplitsTree6
# Run Neighbor-Net algorithm following: phylogenetic_analysis/splitstree_workflow.md
# Export network in Nexus format for further analysis
```

**Visualize and analyze results:**
```bash
# Process SplitsTree6 outputs and generate final visualizations
python phylogenetic_analysis/network_visualization.py --splitstree_output outputs/network.nex


## Results

### 1. Classification Results


### 2. Component Detection
The YOLOv5 component includes a confusion matrix visualization for evaluation of detection performance. This can be found in the yolov5/confusion_matrix directory.
![confusion_matrix](https://github.com/user-attachments/assets/41bfb48c-3812-48f0-9f82-caa0d7f9e9be)

### 3. Phylogenetic Network

## License

This project is licensed under the MIT License - see the LICENSE file for details.

##

