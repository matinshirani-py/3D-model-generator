# 🏥 3D Body Model Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated pipeline for generating **high-fidelity 3D human body models** from anthropometric measurements and medical data using the SMPL-X statistical body model.

## 🌟 Features

- **📊 Medical Data Processing**: Convert patient measurements into body model parameters
- **🎨 3D Model Generation**: Create detailed 3D meshes using SMPL-X
- **⚙️ Multiple Formats**: Export as OBJ (editable) or GLB (web-ready)
- **🚀 GPU Acceleration**: Optional CUDA support for faster processing
- **📈 Medical Analytics**: Calculate BMI, body fat percentage, body type classification
- **👥 Gender Support**: Male, female, and neutral body models

- 
## 📦 SMPL-X Model Files

This project uses the **SMPL-X** body model developed by the Max Planck Institute (MPI).  
To run the pipeline, you must manually download the SMPL-X `.npz` files and place them in the correct folder.

---

### 🔽 Download SMPL-X Model (Required)

You can download the official SMPL-X models from:

🔗 **SMPL-X Official Download:**  
https://smpl-x.is.tue.mpg.de/download.php

You must create a free account, accept the license terms, and download:

- `SMPLX_MALE.npz`  
- `SMPLX_FEMALE.npz`  
- `SMPLX_NEUTRAL.npz`

---
## 📁 Project Structure

```
3d-body-model-generator/
project-root/
│
├── data/
│   ├── inputs/
│   │   ├── patient_001.json
|   |
│   │
│   ├── models/
│   │   └── smplx/
│   │       ├── SMPLX_FEMALE.npz
│   │       ├── SMPLX_MALE.npz
│   │       ├── SMPLX_NEUTRAL.npz
│   │       │
│   │       └── SMPLX_1.1/
│   │           └── smplx/
│   │               ├── SMPLX_FEMALE.npz
│   │               ├── SMPLX_MALE.npz
│   │               └── SMPLX_NEUTRAL.npz
│   │
│   └── outputs/
│       ├── CHILD_CHUBBY_001_summary.json
│       ├── patient_001_summary.txt
│       ├── patient_003_summary.txt
│       ├── patient_004_summary.txt
│       ├── patient_005_summary.txt
│       ├── patient_006_summary.txt
│       └── patient_007_summary.txt
│
│       ├── meshes/
│       │   ├── glb/
│       │   ├── obj/
│       │   │   ├── patient_007_20251204_163623.obj
│       │   │   └── patient_007_20251204_163623_metadata.json
│       │   │
│       │   └── vertices/
|       |
│       │
│       └── parameters/
|
│
├── src/
│   ├── data_processor.py
│   ├── model_generator.py
│   ├── __init__.py
│   
|
│
└── __pycache__/
    └── open3d.cpython-310.pyc

```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/3d-body-model-generator.git
cd 3d-body-model-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SMPL-X models (required)
# Place in data/models/smplx/
# Download from: https://smpl-x.is.tue.mpg.de/
```

### 2. Prepare Patient Data

Create a JSON file in `data/inputs/`:

```json
{
  "patient_id": {
    "value": "P001",
    "description": "Unique patient identifier."
  },
  "name": {
    "value": "Male 44 years old",
    "description": "Patient name or descriptor."
  },
  "height": {
    "value": 171.0,
    "description": "Height in centimeters."
  },
  "weight": {
    "value": 110.0,
    "description": "Weight in kilograms."
  },
  "age": {
    "value": 44,
    "description": "Age of the person in years."
  },
  "gender": {
    "value": "male",
    "description": "Biological sex of the patient."
  },

  "neck": {
    "value": 38.0,
    "description": "Neck circumference in cm."
  },
  "chest": {
    "value": 101.0,
    "description": "Chest circumference in cm."
  },
  "waist": {
    "value": 95.0,
    "description": "Waist circumference in cm."
  },
  "abdomen": {
    "value": 150.0,
    "description": "Abdomen circumference in cm."
  },
  "hips": {
    "value": 110.0,
    "description": "Hip circumference in cm."
  },

  "shoulder_width": {
    "value": 36.0,
    "description": "Shoulder width in cm."
  },

  "upper_arm_right": {
    "value": 32.0,
    "description": "Right upper arm circumference in cm."
  },
  "upper_arm_left": {
    "value": 32.0,
    "description": "Left upper arm circumference in cm."
  },

  "forearm_right": {
    "value": 27.0,
    "description": "Right forearm circumference in cm."
  },
  "forearm_left": {
    "value": 27.0,
    "description": "Left forearm circumference in cm."
  },

  "wrist_right": {
    "value": 17.0,
    "description": "Right wrist circumference in cm."
  },
  "wrist_left": {
    "value": 17.0,
    "description": "Left wrist circumference in cm."
  },

  "arm_length_right": {
    "value": 73.0,
    "description": "Right arm length in cm."
  },
  "arm_length_left": {
    "value": 73.0,
    "description": "Left arm length in cm."
  },

  "leg_length_right": {
    "value": 85.0,
    "description": "Right leg length in cm."
  },
  "leg_length_left": {
    "value": 85.0,
    "description": "Left leg length in cm."
  },

  "thigh_right": {
    "value": 55.0,
    "description": "Right thigh circumference in cm."
  },
  "thigh_left": {
    "value": 55.0,
    "description": "Left thigh circumference in cm."
  },

  "calf_right": {
    "value": 37.0,
    "description": "Right calf circumference in cm."
  },
  "calf_left": {
    "value": 37.0,
    "description": "Left calf circumference in cm."
  },

  "dominant_hand": {
    "value": "right",
    "description": "Dominant hand of the patient."
  },
  "dominant_foot": {
    "value": "right",
    "description": "Dominant foot of the patient."
  },

  "measurement_date": {
    "value": "2024-01-15",
    "description": "Date when measurements were taken."
  },

  "notes": {
    "value": "Patient with abdominal obesity. Medical measurements taken by professional.",
    "description": "Additional notes regarding the patient."
  }
}


### 3. Generate 3D Model

```bash
# Basic usage
python main.py data/inputs/patient_001.json

# With GPU acceleration
python main.py data/inputs/patient_001.json --device cuda

# Export as GLB format
python main.py data/inputs/patient_001.json --format glb

# Custom output directory
python main.py data/inputs/patient_001.json --output-dir results/

# Skip data processing (use existing parameters)
python main.py data/inputs/patient_001.json --no-process
```

## 🎨 Interactive 3D Model Preview

<div class="sketchfab-embed-wrapper"> 
    <iframe title="patient_007_20251204_163623" 
            frameborder="0" 
            allowfullscreen 
            mozallowfullscreen="true" 
            webkitallowfullscreen="true" 
            allow="autoplay; fullscreen; xr-spatial-tracking" 
            xr-spatial-tracking 
            execution-while-out-of-viewport 
            execution-while-not-rendered 
            web-share 
            width="100%" 
            height="480"
            src="https://sketchfab.com/models/32b1887888474986a82ad1c3a6f07bd3/embed">
    </iframe> 
    <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;">
        <a href="https://sketchfab.com/3d-models/patient-007-20251204-163623-32b1887888474986a82ad1c3a6f07bd3?utm_medium=embed&utm_campaign=share-popup&utm_content=32b1887888474986a82ad1c3a6f07bd3" 
           target="_blank" 
           rel="nofollow" 
           style="font-weight: bold; color: #1CAAD9;">
            patient_007_20251204_163623
        </a> by 
        <a href="https://sketchfab.com/matinshirani?utm_medium=embed&utm_campaign=share-popup&utm_content=32b1887888474986a82ad1c3a6f07bd3" 
           target="_blank" 
           rel="nofollow" 
           style="font-weight: bold; color: #1CAAD9;">
            matinshirani
        </a> on 
        <a href="https://sketchfab.com?utm_medium=embed&utm_campaign=share-popup&utm_content=32b1887888474986a82ad1c3a6f07bd3" 
           target="_blank" 
           rel="nofollow" 
           style="font-weight: bold; color: #1CAAD9;">
            Sketchfab
        </a>
    </p>
</div>

## 📊 Output Statistics

| Metric          | Value         |
| --------------- | ------------- |
| Vertices        | ~10,475       |
| Faces           | ~20,944       |
| Height Accuracy | ±2 cm         |
| Processing Time | 30-60 seconds |
| File Size (OBJ) | 5-15 MB       |
| File Size (GLB) | 2-8 MB        |

## 🔧 Technical Details

### SMPL-X Model Architecture

This project uses the **SMPL-X (Skinned Multi-Person Linear Model - eXpressive)** model, which represents the human body with:

- **10,475 vertices** and **20,944 faces**
- **54 body pose parameters** (joint rotations)
- **10 shape parameters** (β parameters)
- **10 expression parameters** (face)
- **3 global orientation parameters**

### Medical Indices Calculated

1. **Body Mass Index (BMI)**: Weight/Height²
2. **Body Fat Percentage**: Using BMI and age
3. **Waist-to-Hip Ratio (WHR)**: Waist/Hip
4. **Body Type Classification**: Ectomorph, Mesomorph, Endomorph
5. **Visceral Fat Risk Assessment**: Based on waist circumference

## 📋 Command Line Arguments

```bash
usage: main.py [-h] [--patient-id PATIENT_ID] [--output-dir OUTPUT_DIR]
               [--format {obj,glb}] [--device {cpu,cuda}] [--gender {male,female,neutral}]
               [--no-process] [--verbose] input_json

positional arguments:
  input_json            Input JSON file with patient measurements

optional arguments:
  -h, --help            show this help message and exit
  --patient-id PATIENT_ID
                        Patient ID (default: extracted from filename or JSON)
  --output-dir OUTPUT_DIR
                        Output directory (default: data/outputs)
  --format {obj,glb}    Output 3D model format (default: obj)
  --device {cpu,cuda}   Device to use for model generation (default: cpu)
  --gender {male,female,neutral}
                        Override gender from JSON file
  --no-process          Skip data processing, use existing parameters
  --verbose, -v         Enable verbose logging
```

## 📦 Dependencies

```txt
torch>=1.9.0
numpy>=1.21.0
trimesh>=3.9.0
smplx>=0.1.28
scikit-learn>=1.0.0
pyrender>=0.1.45
opencv-python>=4.5.0
```

## 🏥 Applications

### Medical & Healthcare

- **Surgical Planning**: Pre-operative simulations
- **Prosthetics**: Custom orthotic device design
- **Nutrition**: Body composition tracking
- **Physical Therapy**: Posture analysis

### Fashion & Retail

- **Virtual Try-on**: Clothing fit prediction
- **Custom Tailoring**: Made-to-measure clothing
- **Avatar Creation**: Digital twins for metaverse

### Research & Development

- **Ergonomics**: Workspace design optimization
- **Biomechanics**: Movement simulation
- **AI Training**: Synthetic data generation

## 🔄 Pipeline Workflow

```
1. 📥 Input: Patient JSON data
2. 🔢 Processing: Normalize measurements, calculate medical indices
3. 🧮 Parameter Estimation: Convert to SMPL-X parameters (β, pose, orientation)
4. 🎨 Model Generation: Create 3D mesh using SMPL-X
5. 💾 Export: Save as OBJ/GLB format
6. 📊 Summary: Generate comprehensive report
```

## 📈 Sample Output Files

```
data/outputs/
├── meshes/
│   ├── P001.obj          # 3D mesh (vertices, faces)
│   ├── P001.mtl          # Material properties
│   └── P001.glb          # GLB format (web compatible)
├── parameters/
│   └── P001_params.pkl   # SMPL-X parameters (for reuse)
└── P001_summary.json     # Complete processing summary
```

## 🚢 Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py", "data/inputs/patient.json"]
```

### Local Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
isort src/
```

## Acknowledgments

- **SMPL-X Model**: Developed by Max Planck Institute for Intelligent Systems
- **PyTorch3D**: Facebook AI Research
- **Trimesh**: Python library for 3D mesh processing

## 📞 Support

For issues and questions:
tel : matin_shirani

- 🐛 [GitHub Issues](https://github.com/yourusername/3d-body-model-generator/issues)

---

**Note**: To view the 3D model interactively, upload your generated GLB file to [Sketchfab](https://sketchfab.com) using the provided script in `scripts/upload_to_sketchfab.py`.
