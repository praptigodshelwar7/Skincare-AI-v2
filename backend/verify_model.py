import onnxruntime as ort
import os

def verify_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    try:
        session = ort.InferenceSession(model_path)
        print(f"\n--- Model Metadata: {model_path} ---")
        
        print("\nInputs:")
        for input in session.get_inputs():
            print(f"  Name: {input.name}")
            print(f"  Shape: {input.shape}")
            print(f"  Type: {input.type}")
            
        print("\nOutputs:")
        for output in session.get_outputs():
            print(f"  Name: {output.name}")
            print(f"  Shape: {output.shape}")
            print(f"  Type: {output.type}")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    # Check both potential model locations
    verify_model("backend/skin_model.onnx")
