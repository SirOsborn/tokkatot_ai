import os
import tensorflow as tf
from onnx_tf.backend import prepare
import onnx
from pathlib import Path

def convert_onnx_to_tflite(onnx_path, tflite_path):
    print(f"Converting {onnx_path} to TFLite...")
    
    # 1. Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # 2. Convert ONNX to TensorFlow Rep
    tf_rep = prepare(onnx_model)
    
    # 3. Export to TensorFlow SavedModel
    temp_pb_path = "temp_tf_model"
    tf_rep.export_graph(temp_pb_path)
    
    # 4. Convert SavedModel to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(temp_pb_path)
    
    # Optional: Quantization (often needed for edge accelerators)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # 5. Save TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    ONNX_DIR = Path("outputs/onnx")
    TFLITE_DIR = Path("outputs/tflite")
    TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    
    models_to_convert = [
        "EfficientNetB0_best.onnx",
        "yolov8n.onnx"
    ]
    
    for model_name in models_to_convert:
        onnx_file = ONNX_DIR / model_name
        tflite_file = TFLITE_DIR / model_name.replace(".onnx", ".tflite")
        
        if onnx_file.exists():
            convert_onnx_to_tflite(str(onnx_file), str(tflite_file))
        else:
            print(f"⚠️ Warning: ONNX model not found at {onnx_file}")
