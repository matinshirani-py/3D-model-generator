"""
src/__init__.py
Package initialization
"""

__version__ = "1.0.0"

# Export main classes
from .data_processor import DataProcessor, PatientData, BodyIndices, SMPLXParameters
from .model_generator import ModelGenerator

__all__ = [
    'DataProcessor',
    'PatientData', 
    'BodyIndices',
    'SMPLXParameters',
    'ModelGenerator'
]