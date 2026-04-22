import requests
import base64
import os

def test_health():
    """Test that the API is alive."""
    url = "https://skincare-api-cmnk.onrender.com/"
    print(f"Testing health: {url}")
    try:
        response = requests.get(url, timeout=60)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_predict():
    """Test the skin prediction endpoint with a dummy image."""
    url = "https://skincare-api-cmnk.onrender.com/api/predict-skin"
    
    # Create a dummy image (224x224 solid color)
    from PIL import Image
    import io
    img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    payload = {
        "image_b64": f"data:image/jpeg;base64,{img_b64}",
        "questionnaire": {
            "q1": "1",
            "q2": "0",
            "q3": "1",
            "q4": "0",
            "q5": "1"
        }
    }
    
    print(f"\nTesting predict: {url}")
    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"  Status: {response.status_code}")
        data = response.json()
        print(f"  Response: {data}")
        
        # Validate response structure
        if response.status_code == 200:
            assert "skin_type" in data, "Missing 'skin_type' in response"
            assert "confidence" in data, "Missing 'confidence' in response"
            assert "breakdown" in data, "Missing 'breakdown' in response"
            assert data["skin_type"] != "", "skin_type is empty!"
            print("  ✓ All assertions passed!")
        elif response.status_code == 400:
            print(f"  (Expected: dummy image has no real face)")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("Skincare AI API — Production Test")
    print("=" * 50)
    
    if test_health():
        test_predict()
    else:
        print("\nAPI is not reachable. It may be cold-starting on Render (wait ~30s and retry).")
