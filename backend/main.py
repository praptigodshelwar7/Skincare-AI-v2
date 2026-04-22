from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional, Any, Union
import numpy as np
import cv2
import base64
import io
import os
import re
import pandas as pd
from PIL import Image
from rapidfuzz import fuzz
import onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR
import gc
import traceback

# ── Global Assets ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_session = None  # ONNX InferenceSession
reader = None  # RapidOCR engine

app = FastAPI(title="SkinCare AI API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health Check (Render pings / to verify the service is alive) ──────────
@app.api_route("/", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "ok", "service": "SkinCare AI API", "version": "3.0"}

# ── ONNX Model Constants ────────────────────────────────────────────────────
CLASS_NAMES = ["dry", "normal", "oily"]

# Normalization constants extracted from the Keras Normalization layer.
# The layer computes: (x - mean) / sqrt(variance)
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
NORM_VAR  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)

# ── Load Ingredient Database ────────────────────────────────────────────────
try:
    csv_path = os.path.join(BASE_DIR, "ingredients_db.csv")
    ingredient_df = pd.read_csv(csv_path)
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

# ── Load Assets on Startup ──────────────────────────────────────────────────
@app.on_event("startup")
async def load_assets():
    global model_session, reader

    # --- Load ONNX Skin Model ---
    model_path = os.path.join(BASE_DIR, "skin_model.onnx")
    try:
        if os.path.exists(model_path):
            # Use only CPU and limit threads to save RAM
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model_session = ort.InferenceSession(
                model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            # Log model input details for debugging
            for inp in model_session.get_inputs():
                print(f"  ONNX Input: name={inp.name}, shape={inp.shape}, type={inp.type}")
            print(f"ONNX skin model loaded from {model_path}")
        else:
            print(f"WARNING: Model file {model_path} not found; using mock predictions.")
    except Exception as e:
        print(f"Model load failed: {e}. Using mock predictions.")

    # --- Load RapidOCR ---
    try:
        # Disable text orientation classification (cls) to save RAM
        reader = RapidOCR(is_cls=False)
        print("RapidOCR initialized (cls=False)")
    except Exception as e:
        print(f"OCR initialization failed: {e}")

# ── Schemas ───────────────────────────────────────────────────────────────────
class QuestionnaireData(BaseModel):
    """Accepts questionnaire answers as strings, ints, or booleans.
    The frontend sends 1/0 (numbers), but we normalize everything to strings."""
    q1: Any = ""
    q2: Any = ""
    q3: Any = ""
    q4: Any = ""
    q5: Any = ""

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
    
    # Standardize detection parameters - slightly more relaxed for better recall
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
    
    if len(faces) == 0:
        return None, 0
        
    # Pick the largest face detected (likely the user)
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    
    # Add 15% padding to match the training preprocessing and ensure coverage
    pad = int(w * 0.15)
    y1 = max(0, y - pad)
    y2 = min(img.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(img.shape[1], x + w + pad)
    
    return img[y1:y2, x1:x2], 1.0

def preprocess_for_onnx(face_crop: np.ndarray) -> np.ndarray:
    """Resize and convert the face crop to a float32 tensor for the ONNX model.
    The model's internal Normalization layer handles mean/std subtraction,
    but we need to scale pixels to [0, 1] first (matching training's rescale=1./255)."""
    img = cv2.resize(face_crop, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV is BGR, model expects RGB
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

def is_yes(val) -> bool:
    """Robustly check if a questionnaire answer means 'yes'.
    Handles: 1, "1", True, "true", "yes", "y" and their negatives."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val == 1
    s = str(val).strip().lower()
    return s in ["yes", "y", "true", "1"]

def questionnaire_scoring(q: QuestionnaireData):
    scores = {"oily": 0, "dry": 0, "normal": 0, "combination": 0}
    
    print(f"DEBUG: Raw questionnaire values: q1={q.q1!r}, q2={q.q2!r}, q3={q.q3!r}, q4={q.q4!r}, q5={q.q5!r}")
    
    if is_yes(q.q1): scores["oily"] += 2; scores["combination"] += 1
    if is_yes(q.q2): scores["dry"] += 3
    if is_yes(q.q3): scores["oily"] += 3
    if is_yes(q.q4): scores["dry"] += 1; scores["normal"] += 0.5
    if is_yes(q.q5): scores["combination"] += 4
    
    total = sum(scores.values()) or 1
    result = {k: v/total for k, v in scores.items()}
    print(f"DEBUG: Questionnaire scores: {result}")
    return result

# ── Core Analysis ─────────────────────────────────────────────────────────────
@app.post("/api/predict-skin")
async def predict_skin_type(data: SkinPredictRequest):
    try:
        img = decode_image(data.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    
    face_crop, conf = detect_face(img)
    
    if face_crop is None:
        raise HTTPException(status_code=400, detail="No face detected. Please ensure good lighting and face the camera directly.")

    # Prediction
    if model_session:
        try:
            input_tensor = preprocess_for_onnx(face_crop)
            input_names = [i.name for i in model_session.get_inputs()]
            
            # Build feed dict — always provide the image tensor as the first input
            feeds = {input_names[0]: input_tensor}
            
            # If the model has additional normalization constant inputs, provide them
            if len(input_names) > 1:
                for name in input_names[1:]:
                    nl = name.lower()
                    # Match common normalization layer naming patterns from tf2onnx
                    if any(k in nl for k in ["sub", "mean", "/y"]):
                        feeds[name] = NORM_MEAN
                    elif any(k in nl for k in ["sqrt", "variance", "std", "/x"]):
                        feeds[name] = NORM_VAR
                    else:
                        # Unknown input — try providing the mean as a safe default
                        print(f"WARNING: Unknown model input '{name}', providing NORM_MEAN as fallback")
                        feeds[name] = NORM_MEAN
            
            preds = model_session.run(None, feeds)[0][0]
            
            # Ensure predictions match CLASS_NAMES length
            num_out = len(preds)
            actual_classes = CLASS_NAMES[:num_out]
            cnn_scores = {actual_classes[i]: float(preds[i]) for i in range(len(actual_classes))}
            
            print(f"DEBUG: ONNX raw predictions: {preds}")
            print(f"DEBUG: ONNX Inputs used: {list(feeds.keys())}")
        except Exception as e:
            print(f"Inference Error: {e}")
            traceback.print_exc()
            cnn_scores = {"dry": 0.33, "normal": 0.34, "oily": 0.33}
    else:
        # Mock for development / when model file is missing
        print("WARNING: Using mock predictions (no model loaded)")
        cnn_scores = {"dry": 0.2, "normal": 0.3, "oily": 0.5}

    q_scores = questionnaire_scoring(data.questionnaire)
    
    # Hybrid calculation
    final_scores = {}
    
    # Derive a 'combination' score from CNN output
    cnn_comb = (cnn_scores.get("oily", 0) * cnn_scores.get("dry", 0) * 4) + (0.1 if cnn_scores.get("normal", 0) > 0.4 else 0)
    cnn_comb = min(cnn_comb, 1.0)
    
    all_cnn = {**cnn_scores, "combination": cnn_comb}
    
    for ct in ["dry", "normal", "oily", "combination"]:
        # 60% Model, 40% Quiz
        final_scores[ct] = (all_cnn.get(ct, 0) * 0.6) + (q_scores.get(ct, 0) * 0.4)
    
    print(f"DEBUG: CNN scores (with derived combination): {all_cnn}")
    print(f"DEBUG: Quiz scores: {q_scores}")
    print(f"DEBUG: Final weighted scores: {final_scores}")
    
    # Normalize to ensure sum is 1.0 (relative confidence)
    total_val = sum(final_scores.values()) or 1
    final_scores = {k: v/total_val for k, v in final_scores.items()}
    
    skin_type = max(final_scores, key=final_scores.get)
    result = {
        "skin_type": skin_type.capitalize(),
        "confidence": round(float(final_scores.get(skin_type, 0)), 4),
        "breakdown": {k: round(v, 4) for k, v in final_scores.items()}
    }
    print(f"DEBUG: Final result: {result}")
    
    # Free memory
    del img, face_crop
    gc.collect()
    
    return result

@app.post("/api/analyze-ingredients")
async def analyze_ingredients(req: IngredientRequest):
    tokens = []
    if req.image_b64:
        if reader is None:
            return {"error": "OCR engine not initialized. Check server logs."}
            
        try:
            # Decode and resize to save RAM
            img = decode_image(req.image_b64)
            
            # Limit max dimension to 800px for OCR (sufficient for text detection)
            h, w = img.shape[:2]
            max_dim = 800
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
            # Robust unpacking
            ocr_out = reader(img)
            result = ocr_out[0] if ocr_out else None
            
            if result:
                # Each line is [[coords], text, confidence] or [coords, text, confidence]
                for line in result:
                    if len(line) >= 2:
                        tokens.append(str(line[1]))
            
            print(f"DEBUG: OCR extracted {len(tokens)} text segments")
            
            # Immediate cleanup
            del img, ocr_out, result
            gc.collect()
        except Exception as e:
            print(f"OCR Runtime Error: {e}")
            traceback.print_exc()
            return {"error": f"OCR processing failed: {str(e)}"}
    
    if req.manual_text:
        tokens.append(req.manual_text)
        
    full_text = " ".join(tokens).lower()
    user_skin = req.skin_type.lower()
    
    # Accurate matching with ingredient list
    detected = []
    suitable = []
    harmful = []
    neutral = []

    for ing in INGREDIENT_LIST:
        # Check if the ingredient name exists as a whole word or significant fuzzy match
        if ing in full_text or (len(ing) > 4 and fuzz.partial_ratio(ing, full_text) > 90):
            detected.append(ing)
            
            ing_data = INGREDIENT_DATA.get(ing, {"good_for": [], "avoid_for": []})
            
            is_good = any(user_skin in s or s in user_skin for s in ing_data["good_for"])
            is_bad = any(user_skin in s or s in user_skin for s in ing_data["avoid_for"])
            
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
        "detected": detected,
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
