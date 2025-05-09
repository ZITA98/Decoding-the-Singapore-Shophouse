# Singapore Shophouse Architectural Elements Dataset

The dataset for this project is hosted on Google Drive. Use the following link to access all data files: [ Singapore Shophouse Analysis Dataset](https://drive.google.com/drive/folders/1Kz10rJhwFtXx0J0y_RaH_PWIZJha2WfF?usp=drive_link)

## Abstract

This dataset presents a comprehensive collection of annotated architectural elements from traditional Singapore shophouses, a significant cultural heritage representing the fusion of Chinese, Malay, and European architectural influences. The dataset comprises high-resolution on-site photography and Google Street View captures, accompanied by detailed JSON annotations of architectural components. This resource is designed to facilitate computational analysis, architectural heritage preservation research, and computer vision applications in the domain of Southeast Asian vernacular architecture, with particular focus on two and three-story shophouse façades.



## Dataset Structure

The dataset is organized into three primary directories:



```
singapore_shophouse_dataset/
├── Shophouse Label/                  # JSON annotation files
│   ├── CT_01_Amoy St_a_02.json
│   ├── CT_01_Amoy St_a_03.json
│   └── ...
├── Shophouse On-site Image/          # High-resolution on-site photographs
│   ├── CT_01_Amoy St_a_02.jpg
│   ├── CT_01_Amoy St_a_03.jpg
│   └── ...
└── Shophouse Google Street View Image/  # Google Street View captures
    ├── ct_001.jpg
    ├── ld_002.jpg
    └── ...
```

Google Street View imagery facilitates unsupervised/semi-supervised street image classification, and the on-site high-resolution images and corresponding LabelMe annotations are intended for training high-precision deep learning models (YOLOv5) for architectural element detection and segmentation, which can maximize recognition accuracy of shophouse architectural components.

## Data Acquisition Methodology

### 1. On-site Photography

#### Temporal Coverage

* Field photography: Conducted 28 Apr. 2025 （Chinatown/Little India/Kampong Glam) and 15 Nov. 2024 (Boat Quay)

#### Camera System

Field photography conducted using Apple iPhone 13

* Original images converted from HEIC to JPEG format while maintaining 3024 × 4032 pixel resolution，with standardized processing protocol applied to ensure data consistency

#### Spatial Coverage

Data collection focused on Singapore's historically significant conservation districts, including:

* Chinatown&#x20;

* Little India&#x20;

* Kampong Glam&#x20;

* Boat Quay&#x20;

### 2. Google Street View (GSV) Data Collection

#### Temporal Coverage

* GSV imagery: Acquired Aug. 2024

#### Technical Parameters

* **API Version:** Google Street View Static API (V2)

* **Resolution:** 640 × 640 pixels

* **Image Format:** RGB, JPEG

* **Viewing Angle:** Front-facing facade views (pitch = 7°, roll = 0°)

* **Field of View (FOV):** 110 degrees

* **Heading:** Perpendicular to building facade (heading = 90° for north-facing facades)

## Annotation Schema

The annotations are created using LabelMe and stored in JSON format with hierarchical structure representing architectural elements. Each shophouse image corresponds to a JSON file containing boundary information and semantic categorization of identified elements on two and three-story shophouse façades.

###



##

