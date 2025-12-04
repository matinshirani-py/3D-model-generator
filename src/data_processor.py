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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PatientData:
    """Patient data class"""
    # Basic information
    patient_id: str = "P0001"
    name: str = "John Doe"
    height: float = 0.0  # cm
    weight: float = 0.0  # kg
    age: int = 0
    gender: str = "male"  # male, female, neutral
    
    # Circumference measurements
    neck: float = 0.0
    chest: float = 0.0
    waist: float = 0.0
    abdomen: float = 0.0
    hips: float = 0.0
    shoulder_width: float = 0.0
    
    # Arm measurements
    upper_arm_right: float = 0.0
    upper_arm_left: float = 0.0
    forearm_right: float = 0.0
    forearm_left: float = 0.0
    wrist_right: float = 0.0
    wrist_left: float = 0.0
    arm_length_right: float = 0.0
    arm_length_left: float = 0.0
    
    # Leg measurements
    leg_length_right: float = 0.0
    leg_length_left: float = 0.0
    thigh_right: float = 0.0
    thigh_left: float = 0.0
    calf_right: float = 0.0
    calf_left: float = 0.0
    
    # Additional info
    dominant_hand: str = "right"
    dominant_foot: str = "right"
    measurement_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: str) -> None:
        """Save to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'PatientData':
        """Load from JSON file with flexible key mapping"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create mapping dictionary
        mapped_data = {}
        
        # Define key mappings
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
        
        # Try to map each field
        for field_name, possible_keys in key_mappings.items():
            value_found = False
            for key in possible_keys:
                if key in data:
                    mapped_data[field_name] = data[key]
                    logger.debug(f"Mapped '{key}' to '{field_name}': {data[key]}")
                    value_found = True
                    break
            
            if not value_found:
                # Try nested structure
                if 'patient_data' in data and isinstance(data['patient_data'], dict):
                    patient_data = data['patient_data']
                    for key in possible_keys:
                        if key in patient_data:
                            mapped_data[field_name] = patient_data[key]
                            logger.debug(f"Mapped nested '{key}' to '{field_name}': {patient_data[key]}")
                            value_found = True
                            break
                
                if not value_found:
                    # Try measurements structure
                    if 'measurements' in data and isinstance(data['measurements'], dict):
                        measurements = data['measurements']
                        for key in possible_keys:
                            if key in measurements:
                                mapped_data[field_name] = measurements[key]
                                logger.debug(f"Mapped measurement '{key}' to '{field_name}': {measurements[key]}")
                                value_found = True
                                break
        
        # Set defaults for required fields if not found
        if 'patient_id' not in mapped_data:
            # Use filename as patient_id
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            mapped_data['patient_id'] = base_name
        
        if 'name' not in mapped_data:
            mapped_data['name'] = mapped_data.get('patient_id', 'Unknown')
        
        # Log what we found
        logger.info(f"Loaded data for patient: {mapped_data.get('name')} (ID: {mapped_data.get('patient_id')})")
        logger.info(f"  Height: {mapped_data.get('height', 'Not found')}cm")
        logger.info(f"  Weight: {mapped_data.get('weight', 'Not found')}kg")
        logger.info(f"  Gender: {mapped_data.get('gender', 'Not found')}")
        
        return cls(**mapped_data)

@dataclass
class BodyIndices:
    """Calculated body indices"""
    bmi: float = 0.0
    body_fat_percentage: float = 0.0
    waist_to_hip_ratio: float = 0.0
    waist_to_height_ratio: float = 0.0
    chest_to_waist_ratio: float = 0.0
    body_type: str = ""  # apple, pear, balanced
    weight_status: str = ""  # underweight, normal, overweight, obese
    visceral_fat_risk: str = ""  # low, moderate, high
    bsa: float = 0.0  # Body Surface Area
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class SMPLXParameters:
    """SMPLX model parameters"""
    betas: np.ndarray  # Shape: (1, 10)
    body_pose: np.ndarray  # Shape: (1, 63)
    global_orient: np.ndarray  # Shape: (1, 3)
    expression: np.ndarray  # Shape: (1, 10)
    device: str = "cpu"
    model_type: str = "smplx"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
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
        """Convert to PyTorch tensors"""
        return {
            'betas': torch.tensor(self.betas, dtype=torch.float32).to(device),
            'body_pose': torch.tensor(self.body_pose, dtype=torch.float32).to(device),
            'global_orient': torch.tensor(self.global_orient, dtype=torch.float32).to(device),
            'expression': torch.tensor(self.expression, dtype=torch.float32).to(device)
        }
    
    def save(self, filepath: str) -> None:
        """Save parameters to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SMPLXParameters':
        """Load parameters from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SMPLXParameters':
        """Create from dictionary"""
        return cls(
            betas=np.array(data['betas'], dtype=np.float32),
            body_pose=np.array(data['body_pose'], dtype=np.float32),
            global_orient=np.array(data['global_orient'], dtype=np.float32),
            expression=np.array(data['expression'], dtype=np.float32),
            device=data.get('device', 'cpu'),
            model_type=data.get('model_type', 'smplx')
        )

class DataProcessor:
    """Main data processing class"""
    
    def __init__(self, output_dir: str = "data/outputs/parameters"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"DataProcessor initialized. Output directory: {output_dir}")
    
    def load_patient_data(self, input_path: str) -> PatientData:
        """Load patient data from JSON file"""
        try:
            logger.info(f"Loading patient data from: {input_path}")
            patient_data = PatientData.from_json(input_path)
            logger.info(f" Patient data loaded: {patient_data.name} (ID: {patient_data.patient_id})")
            
            # Log important fields
            if patient_data.height > 0:
                logger.info(f"  Height: {patient_data.height}cm")
            else:
                logger.warning(f"  Height not found or invalid: {patient_data.height}")
                
            if patient_data.weight > 0:
                logger.info(f"  Weight: {patient_data.weight}kg")
            else:
                logger.warning(f"  Weight not found or invalid: {patient_data.weight}")
                
            logger.info(f"  Gender: {patient_data.gender}")
            
            return patient_data
            
        except FileNotFoundError:
            logger.error(f" File not found: {input_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f" Invalid JSON file: {e}")
            raise
        except Exception as e:
            logger.error(f" Error loading data: {e}")
            raise
    
    def calculate_body_indices(self, patient_data: PatientData) -> BodyIndices:
        """Calculate body indices"""
        logger.info(" Calculating body indices...")
        
        indices = BodyIndices()
        
        # Check if we have valid height and weight
        if patient_data.height <= 0 or patient_data.weight <= 0:
            logger.warning(f" Invalid height ({patient_data.height}cm) or weight ({patient_data.weight}kg)")
            logger.warning("  Cannot calculate BMI and other indices")
            return indices
        
        # Calculate BMI
        height_m = patient_data.height / 100
        indices.bmi = patient_data.weight / (height_m ** 2)
        
        # Body ratios (only if we have measurements)
        if patient_data.waist > 0 and patient_data.hips > 0:
            indices.waist_to_hip_ratio = patient_data.waist / patient_data.hips
        else:
            indices.waist_to_hip_ratio = 0.0
            logger.warning("  Cannot calculate waist-to-hip ratio (missing measurements)")
        
        if patient_data.waist > 0 and patient_data.height > 0:
            indices.waist_to_height_ratio = patient_data.waist / patient_data.height
        else:
            indices.waist_to_height_ratio = 0.0
        
        if patient_data.chest > 0 and patient_data.waist > 0:
            indices.chest_to_waist_ratio = patient_data.chest / patient_data.waist
        else:
            indices.chest_to_waist_ratio = 0.0
        
        # Body type classification
        if indices.waist_to_hip_ratio > 0:
            if indices.waist_to_hip_ratio > 0.95:
                indices.body_type = "apple"  # Android/abdominal obesity
            elif indices.waist_to_hip_ratio < 0.85:
                indices.body_type = "pear"   # Gynoid/gluteal obesity
            else:
                indices.body_type = "balanced"
        else:
            indices.body_type = "unknown"
        
        # Weight status
        if indices.bmi < 18.5:
            indices.weight_status = "underweight"
        elif 18.5 <= indices.bmi < 25:
            indices.weight_status = "normal"
        elif 25 <= indices.bmi < 30:
            indices.weight_status = "overweight"
        else:
            indices.weight_status = "obese"
        
        # Body fat percentage (Deurenberg formula)
        if patient_data.gender == "male":
            indices.body_fat_percentage = 1.2 * indices.bmi + 0.23 * patient_data.age - 10.8 - 5.4
        else:
            indices.body_fat_percentage = 1.2 * indices.bmi + 0.23 * patient_data.age - 5.4
        
        # Visceral fat risk (only if we have waist measurement)
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
        
        # Body Surface Area (Mosteller formula)
        if patient_data.height > 0 and patient_data.weight > 0:
            indices.bsa = np.sqrt((patient_data.height * patient_data.weight) / 3600)
        else:
            indices.bsa = 0.0
        
        logger.info(f"  BMI: {indices.bmi:.1f}")
        logger.info(f"  Body type: {indices.body_type}")
        logger.info(f"  Weight status: {indices.weight_status}")
        if indices.body_fat_percentage > 0:
            logger.info(f"  Body fat: {indices.body_fat_percentage:.1f}%")
        
        return indices
    
    def convert_to_smplx_parameters(self, patient_data: PatientData, body_indices: BodyIndices) -> SMPLXParameters:
        logger.info("Converting to SMPL-X parameters (Improved v2)...")

        SCALE_FACTOR = 4.0        # 
        REVERSE_SIGN = True       # 

        betas = np.zeros((1, 10), dtype=np.float32)  
        def apply_sign(v):
            return -v if REVERSE_SIGN else v

        height = patient_data.height or 170
        weight = patient_data.weight or 70

        # ----------------------------------------------------
      
        bmi = weight / ((height / 100) ** 2)
        bmi_factor = (bmi - 22) / 8           
        betas[0, 0] = apply_sign(bmi_factor * SCALE_FACTOR)

        # ----------------------------------------------------
    
        waist = patient_data.waist or 80
        abdomen = patient_data.abdomen or waist
        hip = patient_data.hips or 95

       
        belly_ratio = (abdomen - waist + 5) / 10   
        betas[0, 1] = apply_sign(-belly_ratio * SCALE_FACTOR * 1.2)

        # ----------------------------------------------------
        # Beta 2 – Upper body muscularity / arm thickness
        arm_r = patient_data.upper_arm_right or 30
        arm_l = patient_data.upper_arm_left or 30
        arm_avg = (arm_r + arm_l) / 2
        arm_excess = (arm_avg - 30) / 5
        betas[0, 2] = apply_sign(-arm_excess * SCALE_FACTOR * 1.4)

        # Beta 3 – Lower body (thighs)
        thigh_r = patient_data.thigh_right or 50
        thigh_l = patient_data.thigh_left or 50
        thigh_avg = (thigh_r + thigh_l) / 2
        thigh_excess = (thigh_avg - 52) / 6
        betas[0, 3] = apply_sign(-thigh_excess * SCALE_FACTOR)

        # ----------------------------------------------------
        # Beta 5 – Shoulder / chest broadness
        shoulder = patient_data.shoulder_width or 42
        chest = patient_data.chest or 95
        shoulder_factor = (shoulder - 42) / 5 + (chest - 95) / 15
        betas[0, 5] = apply_sign(-shoulder_factor * SCALE_FACTOR)

        # Beta 6 – Hip width
        hip_factor = (hip - 95) / 10
        betas[0, 6] = apply_sign(-hip_factor * SCALE_FACTOR)

        # ----------------------------------------------------
      
        if body_indices.body_type == "apple":
            betas[0, 8] = apply_sign(-1.2 * SCALE_FACTOR)   
            betas[0, 9] = apply_sign(+0.8 * SCALE_FACTOR)  
        elif body_indices.body_type == "pear":
            betas[0, 8] = apply_sign(+0.8 * SCALE_FACTOR)
            betas[0, 9] = apply_sign(-1.4 * SCALE_FACTOR)
        
        # ----------------------------------------------------
      
        betas = np.clip(betas, -5.0, 5.0)

      
        active = np.where(np.abs(betas[0]) > 0.4)[0]
        for i in active:
            val = betas[0, i]
            effect = "INCREASES" if val < 0 else "DECREASES"
            logger.info(f"  Beta[{i:2d}] = {val:+5.2f} → {effect} feature")

       
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

        logger.info(f"SMPL-X parameters created | Beta range: [{betas.min():.2f}, {betas.max():.2f}]")
        return params
    
    def process_patient(self, input_json_path: str, 
                       output_params_path: Optional[str] = None,
                       output_report_path: Optional[str] = None) -> Tuple[SMPLXParameters, BodyIndices, PatientData]:
        """Complete processing pipeline for a single patient"""
        logger.info("=" * 60)
        logger.info("PATIENT DATA PROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Load patient data
        patient_data = self.load_patient_data(input_json_path)
        
        # Step 2: Calculate body indices
        body_indices = self.calculate_body_indices(patient_data)
        
        # Step 3: Convert to SMPLX parameters
        smplx_params = self.convert_to_smplx_parameters(patient_data, body_indices)
        
        # Step 4: Save outputs
        if output_params_path is None:
            output_params_path = os.path.join(
                self.output_dir, 
                f"{patient_data.patient_id}_params.pkl"
            )
        
        if output_report_path is None:
            output_report_path = os.path.join(
                self.output_dir,
                f"{patient_data.patient_id}_report.json"
            )
        
        # Save parameters
        smplx_params.save(output_params_path)
        logger.info(f" Parameters saved: {output_params_path}")
        
        # Save comprehensive report
        report = {
            'patient_data': patient_data.to_dict(),
            'body_indices': body_indices.to_dict(),
            'smplx_parameters': smplx_params.to_dict(),
            'processing_timestamp': datetime.now().isoformat(),
            'processor_version': '1.1.0'
        }
        
        with open(output_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Report saved: {output_report_path}")
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        
        return smplx_params, body_indices, patient_data

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = DataProcessor()
    
    # Example usage
    try:
        params, indices, patient = processor.process_patient("data/inputs/patient_001.json")
        print(f"\nPatient: {patient.name}")
        print(f"Height: {patient.height}cm")
        print(f"Weight: {patient.weight}kg")
        print(f"BMI: {indices.bmi:.1f}")
    except Exception as e:
        print(f"Error: {e}")