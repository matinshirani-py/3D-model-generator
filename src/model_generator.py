"""
MATIN SHIRANI
src/model_generator.py
Generate 3D body model from SMPLX parameters with height scaling

| Path                                                 | Line(s) | Purpose                                            |
| ---------------------------------------------------- | ------- | -------------------------------------------------- |
| `"data/models/"`                                     | 21      | Base directory for SMPL-X model files              |
| `"data/outputs/meshes"`                              | 22      | Main output directory for generated models         |
| `os.path.join(output_dir, "obj")`                    | 29      | Directory for exported `.obj` files                |
| `os.path.join(output_dir, "vertices")`               | 30      | Directory for saved NumPy vertex arrays (`.npy`)   |
| `os.path.join(output_dir, "glb")`                    | 31      | Directory for exported `.glb` files                |
| `os.path.join(self.model_path, "smplx", ...)`        | 49–54   | SMPL-X model `.npz` or `.pkl` file loading path    |
| `"data/outputs/parameters/{patient_id}_report.json"` | 133–146 | Possible input path for retrieving patient height  |
| `"data/outputs/parameters/{patient_id}.json"`        | 133–146 | Alternate height source file                       |
| `"data/inputs/{patient_id}.json"`                    | 133–146 | Another fallback input file                        |
| `"data/inputs/{patient_id}_patient.json"`            | 133–146 | Another alternate patient JSON path                |
| `os.path.join(self.output_dir, "obj", ...)`          | 239–245 | Save final 3D output as `.obj`                     |
| `os.path.join(self.output_dir, "glb", ...)`          | 246–252 | Save final 3D output as `.glb`                     |
| `os.path.join(self.output_dir, "vertices", ...)`     | 255–258 | Save vertex positions as `.npy`                    |
| `os.path.join(os.path.dirname(saved_path), ...)`     | 327–336 | Save metadata JSON file next to the exported model |

"""

"""
MATIN SHIRANI
src/model_generator.py
Generate 3D body model from SMPLX parameters with height scaling
"""

# ==========================
# PATH 
# ==========================

MODEL_BASE_PATH = "data/models/"
MESH_OUTPUT_BASE = "data/outputs/meshes"

OBJ_DIR = f"{MESH_OUTPUT_BASE}/obj"
VERTICES_DIR = f"{MESH_OUTPUT_BASE}/vertices"
GLB_DIR = f"{MESH_OUTPUT_BASE}/glb"

PATIENT_JSON_PATHS = [
    "data/outputs/parameters/{patient_id}_report.json",
    "data/outputs/parameters/{patient_id}.json",
    "data/inputs/{patient_id}.json",
    "data/inputs/{patient_id}_patient.json"
]


import torch
import smplx
import numpy as np
import trimesh
import os
import json
from typing import Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelGenerator:
    """3D model generation from SMPLX parameters with height scaling"""
    
    def __init__(self, 
                 model_path: str = MODEL_BASE_PATH,
                 output_dir: str = MESH_OUTPUT_BASE,
                 device: str = "cpu"):
        
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = torch.device(device)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(OBJ_DIR, exist_ok=True)
        os.makedirs(VERTICES_DIR, exist_ok=True)
        os.makedirs(GLB_DIR, exist_ok=True)
        
        self.model = None
        self.model_type = None
        self.current_gender = None
        logger.info(f"ModelGenerator initialized on device: {self.device}")
    
    def load_smplx_model(self, gender: str = "neutral") -> bool:
        """Load SMPLX model"""
        if self.model is not None and self.current_gender == gender:
            logger.info(f"SMPLX model for gender '{gender}' is already loaded")
            return True
            
        logger.info(f"Loading SMPLX model for gender: {gender}")
        
        try:
            model_file = os.path.join(self.model_path, "smplx", f"SMPLX_{gender.upper()}.npz")
            logger.info(f"Looking for model file: {model_file}")
            
            if not os.path.exists(model_file):
                logger.error(f"Model file not found: {model_file}")
                model_dir = os.path.join(self.model_path, "smplx")
                if os.path.exists(model_dir):
                    files = os.listdir(model_dir)
                    logger.info(f"Available files in {model_dir}:")
                    for f in files:
                        logger.info(f"  - {f}")
                return False
         
            self.model = smplx.create(
                model_path=model_file,  
                model_type='smplx',
                gender=gender,
                num_betas=10,
                num_expression_coeffs=10,
                use_face_contour=False,
                ext='npz',
                device=self.device
            )
            self.model_type = 'smplx'
            self.current_gender = gender
            logger.info(f" SMPLX model loaded successfully from: {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"SMPLX failed: {e}")
            
            # Try with .pkl file
            try:
                model_file = os.path.join(self.model_path, "smplx", f"SMPLX_{gender.upper()}.pkl")
                logger.info(f"Trying .pkl file: {model_file}")
                
                if not os.path.exists(model_file):
                    logger.error(f".pkl file not found: {model_file}")
                    return False
                
                self.model = smplx.create(
                    model_path=model_file,
                    model_type='smplx',
                    gender=gender,
                    num_betas=10,
                    use_face_contour=False,
                    ext='pkl',
                    device=self.device
                )
                self.model_type = 'smplx'
                self.current_gender = gender
                logger.info(f" SMPLX model loaded from .pkl: {model_file}")
                return True
                
            except Exception as e2:
                logger.error(f"Both .npz and .pkl failed: {e2}")
                self.current_gender = None
                return False
    
    def scale_vertices_to_height(self, vertices: np.ndarray, target_height_cm: float) -> Tuple[np.ndarray, float]:
        """Scale vertices to match target height in centimeters"""
        if target_height_cm <= 0:
            logger.warning(f"Invalid target height: {target_height_cm}cm. Using original vertices.")
            return vertices, 1.0
        
        current_height_m = np.ptp(vertices[:, 1])  # Y-axis range
        target_height_m = target_height_cm / 100.0
        
        if current_height_m <= 0:
            logger.warning(f"Current model height is zero or invalid: {current_height_m}m")
            return vertices, 1.0
        
        scale_factor = target_height_m / current_height_m
        scaled_vertices = vertices * scale_factor
        
        logger.info(f"Height scaling: {current_height_m:.3f}m -> {target_height_m:.3f}m")
        logger.info(f"Scale factor: {scale_factor:.4f}")
        
        return scaled_vertices, float(scale_factor)
    
    def get_patient_height_from_json(self, patient_id: str) -> Optional[float]:
        """Try multiple JSON paths to find patient height"""
        try:
            paths = [p.format(patient_id=patient_id) for p in PATIENT_JSON_PATHS]
            
            for json_path in paths:
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    height_keys = ['height', 'height_cm', 'patient_height', 'stature']
                    
                    for key in height_keys:
                        if key in data:
                            h = float(data[key])
                            if h > 0:
                                logger.info(f"Found height in {json_path}: {key} = {h}cm")
                                return h
                    
                    if 'patient_data' in data and isinstance(data['patient_data'], dict):
                        for key in height_keys:
                            if key in data['patient_data']:
                                h = float(data['patient_data'][key])
                                if h > 0:
                                    logger.info(f"Found height in patient_data: {h}cm")
                                    return h
        
        except Exception as e:
            logger.warning(f"Could not read height from JSON files: {e}")
        
        return None
    
    def numpy_to_python(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, (list, tuple)):
            return [self.numpy_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.numpy_to_python(value) for key, value in obj.items()}
        else:
            return obj
    
    def generate_mesh(self, betas: np.ndarray, 
                     body_pose: np.ndarray = None,
                     global_orient: np.ndarray = None,
                     expression: np.ndarray = None) -> Dict[str, Any]:
        """Generate 3D mesh from parameters"""
        
        logger.info("Generating 3D mesh...")
        
        if self.model is None:
            logger.error("Model not loaded. Call load_smplx_model() first.")
            raise RuntimeError("Model not loaded")
        
        if body_pose is None:
            body_pose = np.zeros((1, 63), dtype=np.float32)
        if global_orient is None:
            global_orient = np.zeros((1, 3), dtype=np.float32)
        if expression is None:
            expression = np.zeros((1, 10), dtype=np.float32)
        
        try:
            betas_tensor = torch.tensor(betas, dtype=torch.float32).to(self.device)
            body_pose_tensor = torch.tensor(body_pose, dtype=torch.float32).to(self.device)
            global_orient_tensor = torch.tensor(global_orient, dtype=torch.float32).to(self.device)
            expression_tensor = torch.tensor(expression, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                output = self.model(
                    betas=betas_tensor,
                    body_pose=body_pose_tensor,
                    global_orient=global_orient_tensor,
                    expression=expression_tensor,
                    return_verts=True,
                    return_joints=True
                )
            
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            joints = output.joints.detach().cpu().numpy().squeeze()
            faces = self.model.faces
            
            current_height_m = float(np.ptp(vertices[:, 1]))
            
            return {
                'vertices': vertices,
                'faces': faces,
                'joints': joints,
                'model_type': self.model_type,
                'current_height_m': current_height_m,
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating mesh: {e}")
            raise
    
    def save_mesh(self, mesh_data: Dict[str, Any], 
                 patient_id: str,
                 format: str = 'obj') -> str:
        """Save mesh to file"""
        
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        if vertices.ndim == 3:
            vertices = vertices.squeeze(0)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{patient_id}_{timestamp}"
        
        if format.lower() == 'obj':
            filepath = os.path.join(OBJ_DIR, f"{filename}.obj")
            mesh.export(filepath)
            logger.info(f"OBJ file saved: {filepath}")
            
        elif format.lower() == 'glb':
            filepath = os.path.join(GLB_DIR, f"{filename}.glb")
            mesh.export(filepath, file_type='glb')
            logger.info(f"GLB file saved: {filepath}")
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        vertices_path = os.path.join(VERTICES_DIR, f"{filename}_vertices.npy")
        np.save(vertices_path, vertices)
        logger.info(f"Vertices saved: {vertices_path}")
        
        return filepath
    
    def calculate_model_statistics(self, mesh_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate statistics from generated mesh"""
        vertices = mesh_data['vertices']
        
        stats = {
            'n_vertices': int(vertices.shape[0]),
            'n_faces': int(mesh_data['faces'].shape[0]),
            'height': float(np.ptp(vertices[:, 1])),
            'width': float(np.ptp(vertices[:, 0])),
            'depth': float(np.ptp(vertices[:, 2])),
        }
        
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=mesh_data['faces'])
            stats['surface_area'] = float(mesh.area)
            stats['volume'] = float(mesh.volume)
        except Exception as e:
            logger.warning(f"Could not calculate advanced statistics: {e}")
            stats['surface_area'] = 0.0
            stats['volume'] = 0.0
        
        return stats
    
    def generate_model(self, 
                      betas: np.ndarray,
                      patient_id: str,
                      gender: str = "male",
                      body_pose: np.ndarray = None,
                      global_orient: np.ndarray = None,
                      expression: np.ndarray = None,
                      target_height_cm: Optional[float] = None,
                      save_format: str = 'obj') -> Dict[str, Any]:
        """Complete model generation pipeline with height scaling"""
        
        logger.info("=" * 60)
        logger.info("3D MODEL GENERATION PIPELINE")
        logger.info(f"Patient ID: {patient_id}")
        logger.info(f"Gender: {gender}")
        logger.info("=" * 60)
        
        if self.model is None or getattr(self, 'current_gender', None) != gender:
            if not self.load_smplx_model(gender=gender):
                raise RuntimeError(f"Failed to load model for gender: {gender}")
        
        mesh_data = self.generate_mesh(betas, body_pose, global_orient, expression)
        
        if target_height_cm is None:
            target_height_cm = self.get_patient_height_from_json(patient_id)
        
        scale_factor = 1.0
        original_height_m = mesh_data.get('current_height_m', 0)
        scaled_height_m = original_height_m
        
        if target_height_cm is not None and target_height_cm > 0:
            original_vertices = mesh_data['vertices'].copy()
            scaled_vertices, scale_factor = self.scale_vertices_to_height(
                original_vertices, 
                target_height_cm
            )
            mesh_data['vertices'] = scaled_vertices
            
            scaled_height_m = float(np.ptp(scaled_vertices[:, 1]))
            
            logger.info(f" Height scaling applied: {target_height_cm}cm")
            logger.info(f"   Original height: {original_height_m:.3f}m")
            logger.info(f"   Scaled height: {scaled_height_m:.3f}m")
            logger.info(f"   Scale factor: {scale_factor:.4f}")
        else:
            logger.warning(" No target height provided. Using SMPLX default height (~1.8-2.0m)")
        
        saved_path = self.save_mesh(mesh_data, patient_id, save_format)
        stats = self.calculate_model_statistics(mesh_data)
        
        metadata = {
            'patient_id': patient_id,
            'model_type': mesh_data['model_type'],
            'gender': gender,
            'generation_timestamp': mesh_data['generation_timestamp'],
            'file_path': saved_path,
            'statistics': stats,
            'scale_factor': float(scale_factor),
            'target_height_cm': float(target_height_cm) if target_height_cm else None,
            'original_height_m': float(original_height_m),
            'scaled_height_m': float(scaled_height_m),
            'generator_version': '2.1.0'
        }
        
        metadata = self.numpy_to_python(metadata)
        
        metadata_path = os.path.join(
            os.path.dirname(saved_path),
            f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_metadata.json"
        )
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Metadata saved: {metadata_path}")
        
        logger.info("=" * 60)
        logger.info("MODEL GENERATION COMPLETE")
        logger.info("=" * 60)
        
        return {
            'mesh_data': mesh_data,
            'saved_path': saved_path,
            'metadata': metadata,
            'statistics': stats
        }


def test_height_scaling():
    """Test height scaling functionality"""
    
    print("=" * 70)
    print("TESTING HEIGHT SCALING")
    print("=" * 70)
    
    generator = ModelGenerator(device="cpu")
    
    if not generator.load_smplx_model(gender="male"):
        print("Failed to load model")
        return
    
    test_betas = np.zeros((1, 10), dtype=np.float32)
    test_heights = [120.0, 150.0, 170.0, 185.0, 200.0]
    
    for height_cm in test_heights:
        print(f"\n🧪 Testing height: {height_cm}cm")
        
        try:
            result = generator.generate_model(
                betas=test_betas,
                patient_id=f"TEST_H{int(height_cm)}",
                gender="male",
                target_height_cm=height_cm,
                save_format='obj'
            )
            
            actual_height_m = result['statistics']['height']
            actual_height_cm = actual_height_m * 100
            
            print(f"  Target: {height_cm:.1f}cm")
            print(f"  Actual: {actual_height_cm:.1f}cm")
            print(f"  Difference: {abs(actual_height_cm - height_cm):.1f}cm")
            
            if abs(actual_height_cm - height_cm) < 1.0:
                print("   PASS: Height matches target")
            else:
                print("   FAIL: Height mismatch")
                
        except Exception as e:
            print(f"   ERROR: {e}")


if __name__ == "__main__":
    test_height_scaling()
