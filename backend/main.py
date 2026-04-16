from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import base64
import io
import os
import re
import pandas as pd
from PIL import Image
from rapidfuzz import fuzz
import tensorflow as tf
from rapidocr_onnxruntime import RapidOCR
import gc
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Memory Optimization ──────────────────────────────────────────────────────
# Prevent TensorFlow from allocating all RAM on startup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    # On CPU, limit threads to reduce memory footprint
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

# ── Global Assets ────────────────────────────────────────────────────────────
model = None
reader = None # Initialize as None to avoid NameError

# ── Memory Optimization ──────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Assets ──────────────────────────────────────────────────────────────
try:
    ingredient_df = pd.read_csv("ingredients_db.csv")
    ingredient_df["name"] = ingredient_df["name"].astype(str).str.lower().str.strip()
    
    # Pre-process suitability columns
    def parse_suitability(val):
        if pd.isna(val) or val == "[]":
            return []
        # Remove brackets, quotes, and split by comma
        cleaned = str(val).replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
        return [item.strip().lower() for item in cleaned.split(",") if item.strip()]

    ingredient_df["good_for"] = ingredient_df["who_is_it_good_for"].apply(parse_suitability)
    ingredient_df["avoid_for"] = ingredient_df["who_should_avoid"].apply(parse_suitability)
    
    INGREDIENT_LIST = ingredient_df["name"].dropna().unique().tolist()
    # Create lookup dictionary for fast matching
    INGREDIENT_DATA = ingredient_df.set_index("name")[["good_for", "avoid_for"]].to_dict("index")
    
    print(f"Loaded {len(INGREDIENT_LIST)} unique ingredients from CSV")
except Exception as e:
    print(f"Could not load CSV: {e}. Using base list.")
    INGREDIENT_LIST = []
    INGREDIENT_DATA = {}

# Load model
model = None
CLASS_NAMES = ["dry", "normal", "oily"]
T_OPT = 1.2  # Temperature scaling constant

@app.on_event("startup")
async def load_assets():
    global model, reader
    try:
        if os.path.exists("skin_model.h5"):
            model = tf.keras.models.load_model("skin_model.h5")
            print("Skin classification model loaded")
    except Exception as e:
        print(f"Model load failed: {e}. Using mock predictions.")

    try:
        # Load RapidOCR (ONNX based, very low RAM compared to EasyOCR)
        reader = RapidOCR()
        print("RapidOCR initialized")
    except Exception as e:
        print(f"OCR initialization failed: {e}")

# ── Schemas ───────────────────────────────────────────────────────────────────
class QuestionnaireData(BaseModel):
    q1: str
    q2: str
    q3: str
    q4: str
    q5: str

class SkinPredictRequest(BaseModel):
    image_b64: str
    questionnaire: QuestionnaireData

class IngredientRequest(BaseModel):
    image_b64: Optional[str] = None
    manual_text: Optional[str] = None
    skin_type: str

# ── Utils ─────────────────────────────────────────────────────────────────────
def decode_image(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def detect_face(img: np.ndarray):
    # Use Haar Cascade for simplicity and speed in demo
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        return None, 0
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w], 1.0

def questionnaire_scoring(q: QuestionnaireData):
    scores = {"oily": 0, "dry": 0, "normal": 0, "combination": 0}
    yes = lambda x: x.lower() in ["yes", "y", "true"]
    
    if yes(q.q1): scores["oily"] += 2; scores["combination"] += 1
    if yes(q.q2): scores["dry"] += 3
    if yes(q.q3): scores["oily"] += 3
    if yes(q.q4): scores["dry"] += 1; scores["normal"] += 0.5 # sensitive/irritated
    if yes(q.q5): scores["combination"] += 4
    
    total = sum(scores.values()) or 1
    return {k: v/total for k, v in scores.items()}

# ── Core Analysis ─────────────────────────────────────────────────────────────
@app.post("/api/predict-skin")
async def predict_skin_type(data: SkinPredictRequest):
    img = decode_image(data.image_b64)
    face_crop, conf = detect_face(img)
    
    if face_crop is None:
        raise HTTPException(status_code=400, detail="No face detected. Please ensure good lighting.")

    # Prediction
    if model:
        input_img = cv2.resize(face_crop, (224, 224))
        input_img = preprocess_input(input_img)
        input_img = np.expand_dims(input_img, axis=0)
        preds = model.predict(input_img, verbose=0)[0]
        cnn_scores = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    else:
        # Mock for development
        cnn_scores = {"dry": 0.2, "normal": 0.3, "oily": 0.5}

    q_scores = questionnaire_scoring(data.questionnaire)
    
    # Hybrid calculation
    final_scores = {}
    
    # Derive a 'combination' score from CNN output
    # Combination is defined by having both oily and dry characteristics
    # We take the relationship between oily and dry/normal to estimate it
    cnn_comb = (cnn_scores.get("oily", 0) * cnn_scores.get("dry", 0) * 4) + (0.1 if cnn_scores.get("normal", 0) > 0.4 else 0)
    cnn_comb = min(cnn_comb, 1.0) # Cap at 1.0
    
    all_cnn = {**cnn_scores, "combination": cnn_comb}
    
    for ct in ["dry", "normal", "oily", "combination"]:
        # Standardize weights: 60% Model, 40% Quiz
        final_scores[ct] = (all_cnn.get(ct, 0) * 0.6) + (q_scores.get(ct, 0) * 0.4)
    
    print(f"DEBUG: Detailed CNN (with derived comb): {all_cnn}")
    print(f"DEBUG: Q Scores: {q_scores}")
    print(f"DEBUG: Final Weighted Scores: {final_scores}")
    
    # Normalize to ensure sum is 1.0 (relative confidence)
    total_val = sum(final_scores.values()) or 1
    final_scores = {k: v/total_val for k, v in final_scores.items()}
    
    skin_type = max(final_scores, key=final_scores.get)
    
    return {
        "skin_type": skin_type.capitalize(),
        "confidence": final_scores[skin_type],
        "breakdown": final_scores
    }

@app.post("/api/analyze-ingredients")
async def analyze_ingredients(data: IngredientRequest):
    tokens = []
    if data.image_b64:
        if reader is None:
            return {"error": "OCR engine not initialized. Check server logs."}
            
        try:
            img = decode_image(data.image_b64)
            result, _ = reader(img)
            if result:
                tokens.extend([line[1] for line in result])
            
            # Manual cleanup to save RAM
            del img
            gc.collect()
        except Exception as e:
            print(f"OCR Runtime Error: {e}")
            return {"error": f"OCR processing failed: {str(e)}"}
    
    if data.manual_text:
        tokens.append(data.manual_text)
        
    full_text = " ".join(tokens).lower()
    user_skin = data.skin_type.lower()
    
    # Accurate matching with ingredient list
    detected = []
    suitable = []
    harmful = []
    neutral = []

    for ing in INGREDIENT_LIST:
        # Check if the ingredient name exists as a whole word or significant fuzzy match
        if ing in full_text or (len(ing) > 4 and fuzz.partial_ratio(ing, full_text) > 90):
            detected.append(ing)
            
            data = INGREDIENT_DATA.get(ing, {"good_for": [], "avoid_for": []})
            
            is_good = any(user_skin in s or s in user_skin for s in data["good_for"])
            is_bad = any(user_skin in s or s in user_skin for s in data["avoid_for"])
            
            if is_bad:
                harmful.append(ing)
            elif is_good:
                suitable.append(ing)
            else:
                neutral.append(ing)

    # De-duplicate while preserving order
    detected = list(dict.fromkeys(detected))
    suitable = list(dict.fromkeys(suitable))
    harmful = list(dict.fromkeys(harmful))
    neutral = list(dict.fromkeys(neutral))

    return {
        "detected_count": len(detected),
        "suitable": suitable,
        "harmful": harmful,
        "neutral": neutral,
        "verdict": "Generally Suitable" if len(harmful) == 0 else "Use with Caution" if len(harmful) < 2 else "Not Recommended"
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
