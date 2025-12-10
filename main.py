"""
MATIN SHIRANI
main.py - Main execution script with height scaling - FINAL VERSION


| Path                        | Line    | Purpose                     |
| --------------------------- | ------- | --------------------------- |
| `"src"`                     | 15–17   | Module directory            |
| `"data/outputs/parameters"` | 89      | Parameters output directory |
| `"data/models/"`            | 133     | SMPL-X model directory      |
| `"data/outputs/meshes"`     | 134     | 3D model output directory   |
| `"data/outputs/parameters"` | 180     | Parameter file lookup       |
| `"data/outputs/"`           | 197     | Summary file output         |
| Metadata path (indirect)    | 173–178 | Depends on result           |

"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)

# Import modules
from data_processor import DataProcessor, PatientData
from model_generator import ModelGenerator

def get_gender_from_json(json_path: str) -> str:
    """Extract gender from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        gender_keys = ['gender', 'sex', 'patient_gender', 'patient_sex', 'جنسیت']
        
        # Search in root
        for key in gender_keys:
            if key in data:
                gender_value = data[key]
                if isinstance(gender_value, (str, int, float)):
                    gender = str(gender_value).lower().strip()
                    gender_mapping = {
                        'm': 'male', 'male': 'male', 'man': 'male', 'مرد': 'male',
                        'f': 'female', 'female': 'female', 'woman': 'female', 'w': 'female', 'زن': 'female',
                        'n': 'neutral', 'neutral': 'neutral', 'unisex': 'neutral',
                        'unknown': 'neutral', 'other': 'neutral'
                    }
                    if gender in gender_mapping:
                        return gender_mapping[gender]
        
        # Search in nested structures
        if 'patient_data' in data and isinstance(data['patient_data'], dict):
            for key in gender_keys:
                if key in data['patient_data']:
                    gender_value = data['patient_data'][key]
                    if isinstance(gender_value, (str, int, float)):
                        gender = str(gender_value).lower().strip()
                        if gender in gender_mapping:
                            return gender_mapping[gender]
        
    except Exception as e:
        print(f" Warning: Could not extract gender from JSON: {e}")
    
    return 'neutral'

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate 3D body model from patient measurements')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input JSON file with patient measurements')
    parser.add_argument('--patient-id', type=str, default=None,
                       help='Patient ID (default: filename without extension)')
    parser.add_argument('--output-format', type=str, default='obj',
                       choices=['obj', 'glb'], help='Output 3D format')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to use (CPU/GPU)')
    parser.add_argument('--gender', type=str, default=None,
                       choices=['male', 'female', 'neutral'],
                       help='Override gender from JSON file')
    parser.add_argument('--height-cm', type=float, default=None,
                       help='Override height in centimeters')
    parser.add_argument('--no-height-scale', action='store_true',
                       help='Disable height scaling (use default model height)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed logs')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode for missing data')
    
    args = parser.parse_args()
    
    # Configure logging based on verbose flag
    import logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get patient ID from filename if not provided
    if args.patient_id is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.patient_id = base_name
    
    print("=" * 70)
    print(f"3D BODY MODEL GENERATION WITH HEIGHT SCALING")
    print(f"Patient ID: {args.patient_id}")
    print(f"Input file: {args.input}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
   
        print("\n[1/4] PROCESSING PATIENT DATA...")
        processor = DataProcessor(output_dir="data/outputs/parameters")
        smplx_params, body_indices, patient_data = processor.process_patient(args.input)
        
        print(f" Data processed successfully")
        print(f"   Name: {patient_data.name}")
        print(f"   Height from file: {patient_data.height}cm")
        print(f"   Weight: {patient_data.weight}kg")
        print(f"   Gender from file: {patient_data.gender}")
        print(f"   BMI: {body_indices.bmi:.1f}")
        print(f"   Body type: {body_indices.body_type}")
        print(f"   Weight status: {body_indices.weight_status}")
        if hasattr(body_indices, 'body_fat_percentage') and body_indices.body_fat_percentage > 0:
            print(f"   Body fat: {body_indices.body_fat_percentage:.1f}%")
        
        # Step 2: Determine final gender and height
        print(f"\n[2/4] DETERMINING MODEL PARAMETERS...")
        
        # Gender determination
        if args.gender:
            # Use command-line override
            gender = args.gender.lower()
            print(f" Gender: Using command-line specified gender: {gender}")
        else:
            # Use gender from patient data
            gender = patient_data.gender
            if gender not in ['male', 'female', 'neutral']:
                print(f" Invalid gender '{gender}' from file, defaulting to 'neutral'")
                gender = 'neutral'
            print(f" Gender: Using gender from patient data: {gender}")
        
        # Height determination
        if args.height_cm:
            # Use command-line override
            height_cm = args.height_cm
            print(f" Height: Using command-line specified height: {height_cm}cm")
        elif args.no_height_scale:
            # Skip height scaling
            height_cm = None
            print(f" Height: Height scaling disabled (using default model height)")
        elif patient_data.height > 0:
            # Use height from patient data
            height_cm = patient_data.height
            print(f" Height: Using height from patient data: {height_cm}cm")
        else:
            # No height available
            height_cm = None
            print(f" Height: No height found in patient data")
            
            # Interactive mode
            if args.interactive:
                while True:
                    try:
                        height_input = input("\n📏 Enter patient height in cm (or press Enter to use default height): ").strip()
                        if not height_input:
                            print("📏 Using default model height (~1.8-2.0m)")
                            break
                        
                        height_cm = float(height_input)
                        if 50 <= height_cm <= 250:
                            print(f"📏 Using interactive input: {height_cm}cm")
                            break
                        else:
                            print(f" Height must be between 50-250 cm. You entered: {height_cm}cm")
                    except ValueError:
                        print(" Please enter a valid number for height")
            else:
                print(f"📏 Using default model height (~1.8-2.0m)")
                print("   Use --height-cm to specify height or --interactive for interactive mode")
        
        # Step 3: Generate 3D model
        print(f"\n[3/4] GENERATING 3D MODEL...")
        generator = ModelGenerator(
            model_path="data/models/",
            output_dir="data/outputs/meshes",
            device=args.device
        )
        
        # Generate model with specified gender and height
        result = generator.generate_model(
            betas=smplx_params.betas,
            patient_id=args.patient_id,
            gender=gender,
            body_pose=smplx_params.body_pose,
            global_orient=smplx_params.global_orient,
            expression=smplx_params.expression,
            target_height_cm=height_cm,
            save_format=args.output_format
        )
        
        print(f"\n 3D model generated successfully")
        print(f"   Saved to: {result['saved_path']}")
        print(f"   Model type: {result['metadata']['model_type']}")
        print(f"   Gender used: {result['metadata']['gender']}")
        
        # Display height scaling info
        metadata = result['metadata']
        if metadata.get('target_height_cm'):
            print(f"   Target height: {metadata['target_height_cm']}cm")
            print(f"   Original height: {metadata['original_height_m']:.3f}m")
            print(f"   Scaled height: {metadata['scaled_height_m']:.3f}m")
            print(f"   Scale factor: {metadata['scale_factor']:.4f}")
        else:
            print(f"   Using default model height: {metadata['original_height_m']:.3f}m")
            print(f"   (Model was NOT scaled to specific height)")
        
        # Step 4: Display results
        print(f"\n[4/4] RESULTS SUMMARY")
        print("-" * 40)
        stats = result['statistics']
        print(f"Model vertices: {stats['n_vertices']:,}")
        print(f"Model faces: {stats['n_faces']:,}")
        print(f"Model height: {stats['height']:.3f}m ({stats['height']*100:.1f}cm)")
        print(f"Model width: {stats['width']:.3f}m")
        print(f"Model depth: {stats['depth']:.3f}m")
        
        if 'volume' in stats and stats['volume'] > 0:
            print(f"Model volume: {stats['volume']:.3f}m³")
        if 'surface_area' in stats and stats['surface_area'] > 0:
            print(f"Surface area: {stats['surface_area']:.3f}m²")
        
        print("\n" + "=" * 70)
        print(" PROCESS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Show where to find files
        print("\n OUTPUT FILES:")
        print(f"   3D Model: {result['saved_path']}")
        
        # Find parameters file
        params_dir = "data/outputs/parameters"
        if os.path.exists(params_dir):
            for file in os.listdir(params_dir):
                if file.endswith('_params.pkl') and patient_data.patient_id in file:
                    print(f"   Parameters: {os.path.join(params_dir, file)}")
                if file.endswith('_report.json') and patient_data.patient_id in file:
                    print(f"   Report: {os.path.join(params_dir, file)}")
        
        # Find metadata file
        model_dir = os.path.dirname(result['saved_path'])
        metadata_pattern = f"{args.patient_id}_*_metadata.json"
        for file in os.listdir(model_dir):
            if file.endswith('_metadata.json') and args.patient_id in file:
                metadata_file = os.path.join(model_dir, file)
                print(f"   Metadata: {metadata_file}")
                break
        
        # Save comprehensive summary file
        summary_path = f"data/outputs/{args.patient_id}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"3D BODY MODEL GENERATION - COMPREHENSIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"PATIENT INFORMATION:\n")
            f.write(f"  Patient ID: {args.patient_id}\n")
            f.write(f"  Name: {patient_data.name}\n")
            f.write(f"  Gender: {gender}\n")
            if height_cm:
                f.write(f"  Target Height: {height_cm:.1f}cm\n")
            else:
                f.write(f"  Target Height: Not specified (used default)\n")
            f.write(f"  Weight: {patient_data.weight}kg\n")
            f.write(f"  Age: {patient_data.age}\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"MODEL STATISTICS:\n")
            f.write(f"  Vertices: {stats['n_vertices']:,}\n")
            f.write(f"  Faces: {stats['n_faces']:,}\n")
            f.write(f"  Height: {stats['height']:.3f}m ({stats['height']*100:.1f}cm)\n")
            f.write(f"  Width: {stats['width']:.3f}m\n")
            f.write(f"  Depth: {stats['depth']:.3f}m\n")
            if 'volume' in stats and stats['volume'] > 0:
                f.write(f"  Volume: {stats['volume']:.3f}m³\n")
            if 'surface_area' in stats and stats['surface_area'] > 0:
                f.write(f"  Surface Area: {stats['surface_area']:.3f}m²\n\n")
            
            if metadata.get('target_height_cm'):
                f.write(f"HEIGHT SCALING:\n")
                f.write(f"  Target Height: {metadata['target_height_cm']}cm\n")
                f.write(f"  Original Height: {metadata['original_height_m']:.3f}m\n")
                f.write(f"  Scaled Height: {metadata['scaled_height_m']:.3f}m\n")
                f.write(f"  Scale Factor: {metadata['scale_factor']:.4f}\n\n")
            
            f.write(f"MEDICAL INDICES:\n")
            f.write(f"  BMI: {body_indices.bmi:.1f}\n")
            f.write(f"  Body Type: {body_indices.body_type}\n")
            f.write(f"  Weight Status: {body_indices.weight_status}\n")
            if hasattr(body_indices, 'body_fat_percentage') and body_indices.body_fat_percentage > 0:
                f.write(f"  Body Fat: {body_indices.body_fat_percentage:.1f}%\n")
            if hasattr(body_indices, 'waist_to_hip_ratio') and body_indices.waist_to_hip_ratio > 0:
                f.write(f"  Waist-to-Hip Ratio: {body_indices.waist_to_hip_ratio:.2f}\n\n")
            
            f.write(f"FILES:\n")
            f.write(f"  3D Model: {result['saved_path']}\n")
            if os.path.exists(params_dir):
                for file in os.listdir(params_dir):
                    if file.endswith('_params.pkl') and patient_data.patient_id in file:
                        f.write(f"  Parameters: {os.path.join(params_dir, file)}\n")
                    if file.endswith('_report.json') and patient_data.patient_id in file:
                        f.write(f"  Report: {os.path.join(params_dir, file)}\n")
            
            metadata_files = [f for f in os.listdir(model_dir) if f.endswith('_metadata.json') and args.patient_id in f]
            if metadata_files:
                f.write(f"  Metadata: {os.path.join(model_dir, metadata_files[0])}\n")
        
        print(f"   Summary: {summary_path}")
        
    except FileNotFoundError as e:
        print(f"\n ERROR: File not found - {e}")
        print("Please check the input file path.")
        sys.exit(1)
        
    except json.JSONDecodeError as e:
        print(f"\n ERROR: Invalid JSON file - {e}")
        print("Please check the JSON format.")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Process interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

