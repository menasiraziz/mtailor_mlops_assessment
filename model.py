import onnxruntime as ort
import numpy as np
from PIL import Image


def load_image( image_path):

    img = Image.open(image_path).convert("RGB")
    img = np.array(img)  # shape: (H, W, 3)
    img = np.transpose(img, (2, 0, 1))  # shape: (3, H, W)
    img = np.expand_dims(img, axis=0)   # shape: (1, 3, H, W)
    img = img.astype(np.uint8)
    return img

class OnnxClassifier:

    def __init__(self, onnx_path, providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_array):
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        return outputs[0]

    def predict_class(self, input_array):
        logits = self.predict(input_array)
        return int(np.argmax(logits, axis=1)[0])


if __name__ == "__main__":
    image_path = "n01667114_mud_turtle.JPEG"
    onnx_path = "classifier_with_preprocessing.onnx"

    img = load_image(image_path)
    classifier = OnnxClassifier(onnx_path)
    pred = classifier.predict_class(img)
    print(f"Predicted class: {pred}")