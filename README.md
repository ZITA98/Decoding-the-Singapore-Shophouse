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
(https://drive.google.com/drive/folders/1Kz10rJhwFtXx0J0y_RaH_PWIZJha2WfF?usp=drive_link)

You can download it manually and place it in the appropriate directories.

## Results

The YOLOv5 component includes a confusion matrix visualization for evaluation of detection performance. This can be found in the yolov5/confusion_matrix directory.
![confusion_matrix](https://github.com/user-attachments/assets/41bfb48c-3812-48f0-9f82-caa0d7f9e9be)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

##

