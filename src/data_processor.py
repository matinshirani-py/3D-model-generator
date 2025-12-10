"""
MATIN SHIRANI
src/data_processor.py - Process input data and convert to model parameters
"""

import json
import numpy as np
import torch
import os
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
import logging
from datetime import datetime
import pickle

BASE_DIR = "data"
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PARAMETERS_DIR = os.path.join(OUTPUTS_DIR, "parameters")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

DEFAULT_INPUT_FILE = os.path.join(INPUTS_DIR, "patient_001.json")
DEFAULT_PARAMETERS_FILE = "patient_params.pkl"
DEFAULT_REPORT_FILE = "patient_report.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PatientData:
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
                    patient_data = data['patient_data']
                    for key in possible_keys:
                        if key in patient_data:
                            mapped_data[field_name] = patient_data[key]
                            value_found = True
                            break
                
                if not value_found:
                    if 'measurements' in data and isinstance(data['measurements'], dict):
                        measurements = data['measurements']
                        for key in possible_keys:
                            if key in measurements:
                                mapped_data[field_name] = measurements[key]
                                value_found = True
                                break
        
        if 'patient_id' not in mapped_data:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            mapped_data['patient_id'] = base_name
        
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
    
    def to_torch(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        return {
            'betas': torch.tensor(self.betas, dtype=torch.float32).to(device),
            'body_pose': torch.tensor(self.body_pose, dtype=torch.float32).to(device),
            'global_orient': torch.tensor(self.global_orient, dtype=torch.float32).to(device),
            'expression': torch.tensor(self.expression, dtype=torch.float32).to(device)
        }
    
    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SMPLXParameters':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SMPLXParameters':
        return cls(
            betas=np.array(data['betas'], dtype=np.float32),
            body_pose=np.array(data['body_pose'], dtype=np.float32),
            global_orient=np.array(data['global_orient'], dtype=np.float32),
            expression=np.array(data['expression'], dtype=np.float32),
            device=data.get('device', 'cpu'),
            model_type=data.get('model_type', 'smplx')
        )

class DataProcessor:
    
    def __init__(self, output_dir: str = PARAMETERS_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_patient_data(self, input_path: str) -> PatientData:
        try:
            patient_data = PatientData.from_json(input_path)
            
            if patient_data.height > 0:
                pass
            else:
                pass
                
            if patient_data.weight > 0:
                pass
            else:
                pass
                
            return patient_data
            
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as e:
            raise
        except Exception as e:
            raise
    
    def calculate_body_indices(self, patient_data: PatientData) -> BodyIndices:
        
        indices = BodyIndices()
        
        if patient_data.height <= 0 or patient_data.weight <= 0:
            return indices
        
        height_m = patient_data.height / 100
        indices.bmi = patient_data.weight / (height_m ** 2)
        
        if patient_data.waist > 0 and patient_data.hips > 0:
            indices.waist_to_hip_ratio = patient_data.waist / patient_data.hips
        else:
            indices.waist_to_hip_ratio = 0.0
        
        if patient_data.waist > 0 and patient_data.height > 0:
            indices.waist_to_height_ratio = patient_data.waist / patient_data.height
        else:
            indices.waist_to_height_ratio = 0.0
        
        if patient_data.chest > 0 and patient_data.waist > 0:
            indices.chest_to_waist_ratio = patient_data.chest / patient_data.waist
        else:
            indices.chest_to_waist_ratio = 0.0
        
        if indices.waist_to_hip_ratio > 0:
            if indices.waist_to_hip_ratio > 0.95:
                indices.body_type = "apple"
            elif indices.waist_to_hip_ratio < 0.85:
                indices.body_type = "pear"
            else:
                indices.body_type = "balanced"
        else:
            indices.body_type = "unknown"
        
        if indices.bmi < 18.5:
            indices.weight_status = "underweight"
        elif 18.5 <= indices.bmi < 25:
            indices.weight_status = "normal"
        elif 25 <= indices.bmi < 30:
            indices.weight_status = "overweight"
        else:
            indices.weight_status = "obese"
        
        if patient_data.gender == "male":
            indices.body_fat_percentage = 1.2 * indices.bmi + 0.23 * patient_data.age - 10.8 - 5.4
        else:
            indices.body_fat_percentage = 1.2 * indices.bmi + 0.23 * patient_data.age - 5.4
        
        if patient_data.waist > 0:
            if patient_data.gender == "male":
                if patient_data.waist > 102:
                    indices.visceral_fat_risk = "high"
                elif patient_data.waist > 94:
                    indices.visceral_fat_risk = "moderate"
                else:
                    indices.visceral_fat_risk = "low"
            else:
                if patient_data.waist > 88:
                    indices.visceral_fat_risk = "high"
                elif patient_data.waist > 80:
                    indices.visceral_fat_risk = "moderate"
                else:
                    indices.visceral_fat_risk = "low"
        else:
            indices.visceral_fat_risk = "unknown"
        
        if patient_data.height > 0 and patient_data.weight > 0:
            indices.bsa = np.sqrt((patient_data.height * patient_data.weight) / 3600)
        else:
            indices.bsa = 0.0
        
        return indices
    
    def convert_to_smplx_parameters(self, patient_data: PatientData, body_indices: BodyIndices) -> SMPLXParameters:

        SCALE_FACTOR = 4.0
        REVERSE_SIGN = True

        betas = np.zeros((1, 10), dtype=np.float32)
        def apply_sign(v):
            return -v if REVERSE_SIGN else v

        height = patient_data.height or 170
        weight = patient_data.weight or 70

        bmi = weight / ((height / 100) ** 2)
        bmi_factor = (bmi - 22) / 8
        betas[0, 0] = apply_sign(bmi_factor * SCALE_FACTOR)

        waist = patient_data.waist or 80
        abdomen = patient_data.abdomen or waist
        hip = patient_data.hips or 95

        belly_ratio = (abdomen - waist + 5) / 10
        betas[0, 1] = apply_sign(-belly_ratio * SCALE_FACTOR * 1.2)

        arm_r = patient_data.upper_arm_right or 30
        arm_l = patient_data.upper_arm_left or 30
        arm_avg = (arm_r + arm_l) / 2
        arm_excess = (arm_avg - 30) / 5
        betas[0, 2] = apply_sign(-arm_excess * SCALE_FACTOR * 1.4)

        thigh_r = patient_data.thigh_right or 50
        thigh_l = patient_data.thigh_left or 50
        thigh_avg = (thigh_r + thigh_l) / 2
        thigh_excess = (thigh_avg - 52) / 6
        betas[0, 3] = apply_sign(-thigh_excess * SCALE_FACTOR)

        shoulder = patient_data.shoulder_width or 42
        chest = patient_data.chest or 95
        shoulder_factor = (shoulder - 42) / 5 + (chest - 95) / 15
        betas[0, 5] = apply_sign(-shoulder_factor * SCALE_FACTOR)

        hip_factor = (hip - 95) / 10
        betas[0, 6] = apply_sign(-hip_factor * SCALE_FACTOR)

        if body_indices.body_type == "apple":
            betas[0, 8] = apply_sign(-1.2 * SCALE_FACTOR)
            betas[0, 9] = apply_sign(+0.8 * SCALE_FACTOR)
        elif body_indices.body_type == "pear":
            betas[0, 8] = apply_sign(+0.8 * SCALE_FACTOR)
            betas[0, 9] = apply_sign(-1.4 * SCALE_FACTOR)

        betas = np.clip(betas, -5.0, 5.0)

        body_pose = np.zeros((1, 63), dtype=np.float32)
        body_pose[0, [0, 3]]   =  0.05
        body_pose[0, [1, 4]]   =  0.10
        body_pose[0, [15, 18]] =  0.05

        global_orient = np.zeros((1, 3), dtype=np.float32)
        expression    = np.zeros((1, 10), dtype=np.float32)

        params = SMPLXParameters(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            expression=expression,
            device="cpu"
        )

        return params
    
    def process_patient(self, input_json_path: str,
                       output_params_path: Optional[str] = None,
                       output_report_path: Optional[str] = None) -> Tuple[SMPLXParameters, BodyIndices, PatientData]:
        
        patient_data = self.load_patient_data(input_json_path)
        
        body_indices = self.calculate_body_indices(patient_data)
        
        smplx_params = self.convert_to_smplx_parameters(patient_data, body_indices)
        
        if output_params_path is None:
            output_params_path = os.path.join(
                self.output_dir,
                f"{patient_data.patient_id}_params.pkl"
            )
        
        if output_report_path is None:
            output_report_path = os.path.join(
                REPORTS_DIR,
                f"{patient_data.patient_id}_report.json"
            )
        
        smplx_params.save(output_params_path)
        
        report = {
            'patient_data': patient_data.to_dict(),
            'body_indices': body_indices.to_dict(),
            'smplx_parameters': smplx_params.to_dict(),
            'processing_timestamp': datetime.now().isoformat(),
            'processor_version': '1.1.0'
        }
        
        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(output_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return smplx_params, body_indices, patient_data

if __name__ == "__main__":
    processor = DataProcessor()
    
    try:
        params, indices, patient = processor.process_patient(DEFAULT_INPUT_FILE)
        print(f"\nPatient: {patient.name}")
        print(f"Height: {patient.height}cm")
        print(f"Weight: {patient.weight}kg")
        print(f"BMI: {indices.bmi:.1f}")
    except Exception as e:
        print(f"Error: {e}")