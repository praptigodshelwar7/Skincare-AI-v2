import requests
import base64
import os

def test_api():
    url = "https://skincare-api-cmnk.onrender.com/api/predict-skin"
    
    # Create a dummy image (black 224x224)
    from PIL import Image
    import io
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    payload = {
        "image_b64": f"data:image/jpeg;base64,{img_b64}",
        "questionnaire": {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}
    }
    
    print(f"Testing URL: {url}")
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
