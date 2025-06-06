import requests
import base64
import argparse
import time
import os
import sys

BEARER_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTNiZGJhZTk4IiwiaWF0IjoxNzQ5MTk4MDM5LCJleHAiOjIwNjQ3NzQwMzl9.PYO8NFVIgfYThLXH30sG-Z-MMm3PojuvAKcQtl7WVO1TC5t4p3G6_VaandSpVNN_0VTGuLE1_eKPKidb-kj-PoiZG4IYc0NvPQYd7ox_3evLQOAMHcKob1AD6xfu4_byznSLz8B8tXnTRFcDTzjCBqDc7-pBhQ7RVcHSKRpvhwqKcScqK0A8mdquiFt497yU8T3ZPAriMpT0KPrZOqVe707U9wNcm2FCAe1H1ql_tPsj5aZHXZw6UHwR4quhccgDuJUpm8waa6uqPx8_Yv2pmAs9cX1wCasWBV2YWOe09TJDiotzStClSCshcCt98aDxwefg5H3IUiraOEIZv43ptg"

HEADERS = {
    'Authorization': f'Bearer {BEARER_TOKEN}',
    'Content-Type': 'application/json'
}

HEALTH_URL = "https://api.cortex.cerebrium.ai/v4/p-3bdbae98/classifier/health"
PREDICT_URL = "https://api.cortex.cerebrium.ai/v4/p-3bdbae98/classifier/predict"

def check_health():
    resp = requests.get(HEALTH_URL, headers=HEADERS)
    print(f"Health check status: {resp.status_code}, response: {resp.text}")
    return resp.status_code == 200 and resp.json().get("status") == "ok"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image file {image_path} does not exist.")
        sys.exit(1)
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(PREDICT_URL, headers=HEADERS, json={"image": img_b64})
    if resp.status_code == 200:
        result = resp.json()
        print(f"Prediction result: {result}")
        print(f"Predicted class id: {result.get('class_id')}")
        return result
    else:
        print(f"Prediction failed: {resp.status_code}, {resp.text}")
        return None

def test_latency(image_path):
    start = time.time()
    predict_image(image_path)
    latency = time.time() - start
    print(f"Prediction latency: {latency:.2f} seconds")
    return latency

def test_invalid_input():
    print("Testing invalid input (empty image)...")
    resp = requests.post(PREDICT_URL, headers=HEADERS, json={"image": ""})
    print(f"Status: {resp.status_code}, Response: {resp.text}")

def test_missing_field():
    print("Testing missing image field...")
    resp = requests.post(PREDICT_URL, headers=HEADERS, json={})
    print(f"Status: {resp.status_code}, Response: {resp.text}")

def run_preset_tests():
    print("Running preset tests...")
    assert check_health(), "Health check failed!"
    sample_image = "n01667114_mud_turtle.JPEG"
    if not os.path.exists(sample_image):
        print(f"Sample image {sample_image} not found. Please provide it for preset tests.")
        return
    test_latency(sample_image)
    test_invalid_input()
    test_missing_field()
    print("Preset tests completed.")

def main():
    parser = argparse.ArgumentParser(description="Test Cerebrium model deployment.")
    parser.add_argument("--image", type=str, help="Path to image file for prediction.")
    parser.add_argument("--preset", action="store_true", help="Run preset custom tests.")
    args = parser.parse_args()

    if args.preset:
        run_preset_tests()
    elif args.image:
        if check_health():
            predict_image(args.image)
        else:
            print("Service is not healthy! Skipping prediction.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()