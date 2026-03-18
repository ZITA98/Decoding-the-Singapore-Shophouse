# Singapore Shophouse Architectural Elements Dataset

## Abstract
This dataset presents a comprehensive collection of annotated architectural 
elements from traditional Singapore shophouses, representing the fusion of 
Chinese, Malay, and European architectural influences. This resource is 
designed to facilitate computational analysis, architectural heritage 
preservation research, and computer vision applications in Southeast Asian 
vernacular architecture, with particular focus on two and three-story 
shophouse façades.

## Dataset Access

| Component | Format | Access | Location |
|-----------|--------|--------|----------|
| JSON annotation files | .json | Public | GitHub: https://github.com/ZITA98/Decoding-the-Singapore-Shophouse |
| Google street view sampling points and coordinates  | .csv | Public | Uploaded with manuscript (Supplementary Data 3)|
| On-site facade photographs | .jpg | Restricted | Zenodo (DOI: [insert DOI]) |

## Notes on Access

### Google Street View Images
Google Street View images cannot be redistributed due to Google's Terms 
of Service. Sampling points and coordinates for all Street View images 
are provided in Supplementary Data 3, enabling retrieval of the 
corresponding imagery via the Google Street View Static API:
https://developers.google.com/maps/documentation/streetview

### On-site Photographs (Restricted)
On-site facade photographs are withheld due to privacy considerations. 
Researchers requiring access may contact the author.

## Data Acquisition

### On-site Photography
- Device: Apple iPhone 13
- Resolution: 3024 × 4032 pixels (HEIC converted to JPEG)
- Chinatown / Little India / Kampong Glam: 28 Apr. 2025
- Boat Quay: 15 Nov. 2024
- Coverage: Chinatown, Little India, Kampong Glam, Boat Quay

### Google Street View
- Acquired: Aug. 2024
- API: Google Street View Static API (V2)
- Resolution: 640 × 640 pixels
- Format: RGB, JPEG
- Pitch: 7°, FOV: 110°, Heading: perpendicular to facade

## Annotation Schema
Annotations were created using LabelMe and stored in JSON format, 
with hierarchical structure representing architectural elements. 
Each image corresponds to a JSON file containing boundary information 
and semantic categorisation of identified elements on two and 
three-story shophouse façades. Annotation files are available via 
the GitHub repository.

