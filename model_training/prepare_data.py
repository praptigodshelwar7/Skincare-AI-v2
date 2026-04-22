import os
import cv2
import numpy as np
from pathlib import Path

def crop_face(image_path, output_path):
    """Detects face and saves a tight crop for training."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # We use a slightly more generous scale factor to find smaller faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Sort by area and pick the largest face
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        # Crop with some padding to include forehead/cheeks
        pad = int(w * 0.1)
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        
        face_crop = img[y1:y2, x1:x2]
        cv2.imwrite(str(output_path), face_crop)
        return True
    return False

def process_dataset(input_dir, output_dir):
    """Walks through dataset and crops all faces into a new structure."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    categories = ['dry', 'normal', 'oily']
    splits = ['train', 'valid', 'test']
    
    count_success = 0
    count_fail = 0
    
    for split in splits:
        for cat in categories:
            src_dir = input_path / split / cat
            dst_dir = output_path / split / cat
            
            if not src_dir.exists():
                print(f"Skipping {src_dir} (not found)")
                continue
                
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing {split}/{cat}...")
            for img_file in src_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    success = crop_face(img_file, dst_dir / img_file.name)
                    if success:
                        count_success += 1
                    else:
                        count_fail += 1
                        
    print(f"\nProcessing Complete!")
    print(f"Successfully cropped: {count_success}")
    print(f"Failed (no face detected): {count_fail}")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    # Updated to match your current folder structure
    process_dataset("model_training/data/raw", "model_training/data/processed")
