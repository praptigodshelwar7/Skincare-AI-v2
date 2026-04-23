# Skincare AI Pro: Intelligent Dermatological Analysis System

## Overview
Skincare AI Pro is a comprehensive web-based application designed to provide users with personalized skincare analysis. The system utilizes state-of-the-art deep learning models to classify skin types and advanced optical character recognition (OCR) to analyze product ingredients for suitability based on the user's unique dermatological profile.

## Key Features
*   **AI-Powered Skin Classification**: Utilizes an EfficientNetB0-based convolutional neural network (CNN) to categorize skin into three primary types: Dry, Normal, and Oily.
*   **Two-Phase Training Strategy**: Implements a robust training pipeline with initial head-training followed by aggressive fine-tuning of the top layers for maximum stability and accuracy.
*   **Hybrid Scoring System**: Combines visual AI analysis with a structured dermatological questionnaire to increase prediction accuracy and account for user-perceived skin behavior.
*   **Automated Ingredient Analysis**: Employs OCR technology to extract ingredient lists from product labels and cross-references them with a proprietary database of ingredients.
*   **Safety & Suitability Verdicts**: Provides instant feedback on whether a product is suitable, should be used with caution, or is not recommended for the user's specific skin type.

## Architecture
The project follows a decoupled client-server architecture:
*   **Frontend**: A responsive Single Page Application (SPA) built with React and Vite.
*   **Backend**: A high-performance REST API built with FastAPI, utilizing ONNX Runtime for efficient model inference.
*   **Machine Learning Pipeline**: Includes advanced face detection (MTCNN), test-time augmentation (TTA), and a fine-tuned EfficientNetB0 architecture.

## Dataset link 

* https://app.roboflow.com/skintype-ssboo/projects
  
## Technology Stack

### Frontend
*   React.js
*   Vite (Build Tool)
*   Axios (API Communication)
*   Lucide React (Iconography)
*   React Webcam (Image Capture)
*   Vanilla CSS (Modern UI/UX)

### Backend
*   FastAPI (Python Framework)
*   Uvicorn (ASGI Server)
*   ONNX Runtime (Machine Learning Inference)
*   RapidOCR (Text Extraction)
*   OpenCV (Image Processing)
*   Pandas (Data Management)

## Installation and Setup

### Prerequisites
*   Python 3.9 or higher
*   Node.js (LTS version)
*   NPM or Yarn

### Backend Configuration
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   python main.py
   ```

### Frontend Configuration
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the root of the frontend folder:
   ```env
   VITE_API_BASE=http://localhost:8000
   ```
4. Launch the development server:
   ```bash
   npm run dev
   ```

## Model Performance
The current skin classification model is based on **EfficientNetB0**, which offers a superior accuracy-to-size ratio compared to older architectures.
*   **Architecture**: EfficientNetB0 (Fine-tuned)
*   **Training Accuracy**: ~87.4% (Phase 2 completion)
*   **Validation Accuracy**: ~84.2%
*   **Inference Engine**: ONNX Runtime (CPU optimized).

## Output and Results
<img width="1315" height="720" alt="image" src="https://github.com/user-attachments/assets/d9703b94-a2f6-46fd-bb82-3e2b392949db" />


## Deployment
The system is configured for deployment on the Render platform.
*   **Service Type**: Web Service (Backend) and Static Site (Frontend).
*   **Configuration**: Refer to `render.yaml` for environment variables and build commands.

## Disclaimer
This application is designed for informational and educational purposes only. It does not provide medical advice, diagnosis, or treatment. Always seek the advice of a dermatologist or other qualified health provider with any questions regarding a medical condition.
