"""
Inference service - wrapper for ensemble model predictions.
"""

import sys
from pathlib import Path

# Setup paths for development imports
project_root = Path(__file__).parent.parent.parent.parent
dev_path = project_root / "development"

# Add development paths
sys.path.insert(0, str(dev_path))
sys.path.insert(0, str(dev_path / "models"))
sys.path.insert(0, str(dev_path / "data_prep"))
sys.path.insert(0, str(project_root))

# Create a simple module alias for 'data' to point to data_utils
import importlib.util
spec = importlib.util.spec_from_file_location("data", dev_path / "data_prep" / "data_utils.py")
data = importlib.util.module_from_spec(spec)
sys.modules['data'] = data
spec.loader.exec_module(data)

# Now import
from models.inference import ChickenDiseaseDetector

__all__ = ["ChickenDiseaseDetector"]
