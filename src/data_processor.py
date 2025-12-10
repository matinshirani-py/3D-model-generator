"""
MATIN SHIRANI
src/data_processor.py - Process input data and convert to model parameters
"""

# ===========================
# PATH CONFIGURATION 
# ==========================*

PARAM_OUTPUT_BASE = "data/outputs/parameters"
INPUT_BASE = "data/inputs"



import json
import numpy as np
import torch
import os
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
import logging
from datetime import datetime
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



@dataclass
class PatientData:
    """Patient data class"""
    patient_id: str = "P0001"
    name: str = "John Doe"
    height: float = 0.0
    weight: float = 0.0
    age: int = 0
    gender: str = "male"

    neck: float = 0.0
    chest: float = 0.0
    waist: float = 0.0
    abdomen: float = 0.0
    hips: float = 0.0
    shoulder_width: float = 0.0

    upper_arm_right: float = 0.0
    upper_arm_left: float = 0.0
    forearm_right: float = 0.0
    forearm_left: float = 0.0
    wrist_right: float = 0.0
    wrist_left: float = 0.0
    arm_length_right: float = 0.0
    arm_length_left: float = 0.0

    leg_length_right: float = 0.0
    leg_length_left: float = 0.0
    thigh_right: float = 0.0
    thigh_left: float = 0.0
    calf_right: float = 0.0
    calf_left: float = 0.0

    dominant_hand: str = "right"
    dominant_foot: str = "right"
    measurement_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath: str) -> 'PatientData':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        mapped_data = {}

        key_mappings = {
            'patient_id': ['patient_id', 'id', 'patientID', 'patientId'],
            'name': ['name', 'patient_name', 'patientName', 'نام'],
            'height': ['height', 'height_cm', 'stature', 'طول', 'قد'],
            'weight': ['weight', 'weight_kg', 'mass', 'وزن'],
            'age': ['age', 'سن'],
            'gender': ['gender', 'sex', 'patient_gender', 'patient_sex', 'جنسیت'],
            'chest': ['chest', 'chest_circumference', 'سینه'],
            'waist': ['waist', 'waist_circumference', 'کمر'],
            'hips': ['hips', 'hip_circumference', 'باسن'],
            'abdomen': ['abdomen', 'abdominal_circumference', 'شکم'],
            'shoulder_width': ['shoulder_width', 'shoulderWidth', 'عرض_شانه'],
            'neck': ['neck', 'neck_circumference', 'گردن'],
        }

        for field_name, possible_keys in key_mappings.items():
            value_found = False
            for key in possible_keys:
                if key in data:
                    mapped_data[field_name] = data[key]
                    value_found = True
                    break

            if not value_found:
                if 'patient_data' in data and isinstance(data['patient_data'], dict):
                    for key in possible_keys:
                        if key in data['patient_data']:
                            mapped_data[field_name] = data['patient_data'][key]
                            value_found = True
                            break

            if not value_found:
                if 'measurements' in data and isinstance(data['measurements'], dict):
                    for key in possible_keys:
                        if key in data['measurements']:
                            mapped_data[field_name] = data['measurements'][key]
                            value_found = True
                            break

        if 'patient_id' not in mapped_data:
            mapped_data['patient_id'] = os.path.splitext(os.path.basename(filepath))[0]
        if 'name' not in mapped_data:
            mapped_data['name'] = mapped_data.get('patient_id', 'Unknown')

        return cls(**mapped_data)


@dataclass
class BodyIndices:
    bmi: float = 0.0
    body_fat_percentage: float = 0.0
    waist_to_hip_ratio: float = 0.0
    waist_to_height_ratio: float = 0.0
    chest_to_waist_ratio: float = 0.0
    body_type: str = ""
    weight_status: str = ""
    visceral_fat_risk: str = ""
    bsa: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SMPLXParameters:
    betas: np.ndarray
    body_pose: np.ndarray
    global_orient: np.ndarray
    expression: np.ndarray
    device: str = "cpu"
    model_type: str = "smplx"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'betas': self.betas.tolist(),
            'body_pose': self.body_pose.tolist(),
            'global_orient': self.global_orient.tolist(),
            'expression': self.expression.tolist(),
            'device': self.device,
            'model_type': self.model_type,
            'timestamp': self.timestamp
        }

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

# ============
# DATA PROCESSOR CLASS
# ===============

class DataProcessor:
    def __init__(self, output_dir: str = PARAM_OUTPUT_BASE):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"DataProcessor initialized. Output directory: {output_dir}")
