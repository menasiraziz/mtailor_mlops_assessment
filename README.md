### PyTorch to ONNX Image Classifier Pipeline

This repository provides a complete workflow for exporting a PyTorch image classifier to ONNX format, including preprocessing, and running inference with ONNX Runtime. It also includes comprehensive tests to ensure correctness and consistency between PyTorch and ONNX outputs.

---

#### Contents

- **convert_to_onnx.py**: Exports a PyTorch model (with preprocessing) to ONNX.
- **model.py**: Loads ONNX model, preprocesses images, and runs inference.
- **test.py**: Runs a suite of tests for PyTorch and ONNX inference, export, and error handling.

---

#### 1. Model Export: `convert_to_onnx.py`

- **PreprocessingWrapper**: Wraps the PyTorch model to include image preprocessing (uint8 to float, resize to 224x224, normalization).
- **Export**: Loads model weights, wraps the model, and exports to ONNX (`classifier_with_preprocessing.onnx`) with dynamic batch and image size support.
- **Input**: Expects images as `uint8` tensors in NCHW format (batch, channels, height, width).
- **Output**: ONNX model with preprocessing embedded, ready for direct inference on raw images.

---

#### 2. ONNX Inference: `model.py`

- **load_image**: Loads and preprocesses an image file into the required NCHW `uint8` format.
- **OnnxClassifier**: Loads the ONNX model and provides:
    - `predict`: Returns raw model outputs (logits).
    - `predict_class`: Returns the predicted class index.
- **Example Usage**: Run the script directly to classify a sample image.

---

#### 3. Testing: `test.py`

- **test_pytorch_forward**: Checks PyTorch model forward pass with preprocessing.
- **test_onnx_export_and_inference**: Exports ONNX, checks model validity, and runs inference.
- **test_consistency**: Verifies that PyTorch and ONNX outputs are numerically close.
- **test_batch_and_dynamic_shapes**: Tests ONNX model with different batch sizes and image resolutions.
- **test_error_handling**: Ensures the ONNX model rejects invalid input types and shapes.
- **test_specific_image_classification**: Checks that a specific image is classified as the expected class.

All tests can be run via `python test.py` (ensure required files are present).

---



#### Usage

1. **Export ONNX Model**  
   Run `convert_to_onnx.py` to generate `classifier_with_preprocessing.onnx`.

2. **Inference**  
   Use `model.py` to classify images with the ONNX model.

3. **Testing**  
   Run `test.py` to verify the pipeline and model correctness.

---


#### File Checklist

- `pytorch_model.py`: Contains the model definition (`Classifier`, `BasicBlock`).
- `pytorch_model_weights.pth`: Trained model weights.
- `n01667114_mud_turtle.JPEG`: Example test image.

---


```bash
# Export ONNX model
python convert_to_onnx.py

# Run ONNX inference
python model.py

# Run all tests
python test.py
```

---

### FastAPI ONNX Image Classifier Service



---

#### Features

- **/predict** endpoint: Accepts a base64-encoded image and returns the predicted class.
- **/health** endpoint: Simple health check for service monitoring.
- **ONNX Runtime**: Efficient inference using a pre-exported ONNX model with preprocessing.
- **Robust error handling**: Handles invalid input, missing fields, and model errors gracefully.
- **Test client**: `test_server.py` for automated and manual endpoint testing.

---

#### File Overview

- **main.py**: FastAPI server exposing `/predict` and `/health` endpoints.
- **test_server.py**: Command-line tool for testing the deployed API, including health checks, prediction, latency, and error cases.

---

#### API Usage

##### 1. Start the Server

Make sure you have the ONNX model file (`classifier_with_preprocessing.onnx`) in the working directory.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

##### 2. `/predict` Endpoint

- **Method**: POST
- **Request Body**: JSON with a single field `image` (base64-encoded image data, e.g. JPEG or PNG).
- **Response**: JSON with the predicted class ID.

**Example Request:**

```json
POST /predict
{
  "image": "<base64-encoded-image>"
}
```

**Example Response:**

```json
{
  "class_id": 35
}
```

**Error Responses:**

- 400: Missing or empty image field, invalid base64, or invalid image data.
- 500: Model prediction failure.

##### 3. `/health` Endpoint

- **Method**: GET
- **Response**: `{ "status": "ok" }` if the service is running.

---

#### Model Input Format

- Expects images as base64-encoded strings in the request.
- Internally, images are converted to NCHW `uint8` arrays (batch, channels, height, width).
- Preprocessing (resize, normalization) is handled inside the ONNX model.

---

#### Testing the API

Use the provided `test_server.py` script to test the deployed API.



**Example Usage:**

- **Health Check:**
  ```bash
  python test_server.py --preset
  ```
- **Predict a Single Image:**
  ```bash
  python test_server.py --image path/to/image.jpg
  ```
- **Run Preset Tests (health, latency, invalid input):**
  ```bash
  python test_server.py --preset
  ```



#### Error Handling

- Returns clear error messages for missing/invalid input and model errors.
- Handles invalid base64, non-image data, and wrong input shapes/types.

---


Install dependencies with:

```bash
pip install fastapi uvicorn onnxruntime numpy pillow
```

---



