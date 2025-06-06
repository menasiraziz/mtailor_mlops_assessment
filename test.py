import os
import torch

import numpy as np
import onnx

from pytorch_model import Classifier, BasicBlock
from model import load_image
from model import OnnxClassifier
from convert_to_onnx import PreprocessingWrapper

def test_pytorch_forward(model, image_path):
    img = load_image(image_path)
    img_tensor = torch.from_numpy(img)
    wrapper = PreprocessingWrapper(model)
    wrapper.eval()
    with torch.no_grad():
        out = wrapper(img_tensor)
    assert out.shape[0] == 1, "Batch size mismatch"
    print("PyTorch forward pass OK. Output shape:", out.shape)
    return out

def test_onnx_export_and_inference(model, image_path, onnx_path="classifier_with_preprocessing.onnx"):
    wrapper = PreprocessingWrapper(model)
    wrapper.eval()
    dummy_input = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8)
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"}, "output": {0: "batch_size"}},
    )
    print("ONNX export OK:", onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    img = load_image(image_path)
    classifier = OnnxClassifier(onnx_path)
    logits = classifier.predict(img)
    assert logits.shape[0] == 1, "ONNX batch size mismatch"
    print("ONNX inference OK. Output shape:", logits.shape)
    return logits

def test_specific_image_classification(onnx_path, image_path, expected_class=35):
    img = load_image(image_path)
    classifier = OnnxClassifier(onnx_path)
    pred = classifier.predict_class(img)
    print(f"Predicted class: {pred}")
    assert pred == expected_class, f"Expected class {expected_class}, but got {pred}"
    print(f"Output class is {expected_class} as expected.")

def test_consistency(model, image_path, onnx_path):
    img = load_image(image_path)
    img_tensor = torch.from_numpy(img)
    wrapper = PreprocessingWrapper(model)
    wrapper.eval()
    with torch.no_grad():
        torch_out = wrapper(img_tensor).numpy()
    classifier = OnnxClassifier(onnx_path)
    onnx_out = classifier.predict(img)
    np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-3)
    print("PyTorch and ONNX outputs are consistent.")

def test_batch_and_dynamic_shapes(model, onnx_path):
    classifier = OnnxClassifier(onnx_path)
    batch_imgs = np.random.randint(0, 256, (4, 3, 300, 300), dtype=np.uint8)
    preds = classifier.predict(batch_imgs)
    assert preds.shape[0] == 4, "ONNX batch inference failed"
    print("ONNX batch inference OK.")
    for h, w in [(128,128), (512,512), (224,224)]:
        img = np.random.randint(0, 256, (1, 3, h, w), dtype=np.uint8)
        out = classifier.predict(img)
        assert out.shape[0] == 1, f"ONNX dynamic shape failed for {h}x{w}"
    print("ONNX dynamic input shape OK.")

def test_error_handling(onnx_path):
    classifier = OnnxClassifier(onnx_path)
    try:
        img = np.random.rand(1,3,224,224).astype(np.float32)
        classifier.predict(img)
        print("ERROR: Model should not accept float32 input!")
    except Exception as e:
        print("Correctly caught error for wrong dtype:", str(e))
    try:
        img = np.random.randint(0,256,(1,224,224,3),dtype=np.uint8)
        classifier.predict(img)
        print("ERROR: Model should not accept NHWC input!")
    except Exception as e:
        print("Correctly caught error for wrong shape:", str(e))

if __name__ == "__main__":
    image_path = "n01667114_mud_turtle.JPEG"
    onnx_path = "classifier_with_preprocessing.onnx"
    assert os.path.exists("pytorch_model_weights.pth"), "Missing model weights"
    assert os.path.exists("pytorch_model.py"), "Missing model code"
    assert os.path.exists(image_path), f"Missing test image: {image_path}"

    model = Classifier(BasicBlock, [2,2,2,2])
    model.load_state_dict(torch.load("pytorch_model_weights.pth", map_location='cpu'))
    model.eval()

    print("==== Testing PyTorch forward ====")
    test_pytorch_forward(model, image_path)

    print("==== Testing ONNX export and inference ====")
    test_onnx_export_and_inference(model, image_path, onnx_path)

    print("==== Testing PyTorch/ONNX consistency ====")
    test_consistency(model, image_path, onnx_path)

    print("==== Testing ONNX batch and dynamic shapes ====")
    test_batch_and_dynamic_shapes(model, onnx_path)

    print("==== Testing ONNX error handling ====")
    test_error_handling(onnx_path)

    print("==== Testing ONNX for specific image ====")

    test_specific_image_classification(onnx_path, image_path, expected_class=35)

    print("==== All tests passed! ====")