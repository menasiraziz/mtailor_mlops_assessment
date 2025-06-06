import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_model import Classifier, BasicBlock

class PreprocessingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(x)

model = Classifier(BasicBlock, [2,2,2,2])
model.load_state_dict(torch.load("./pytorch_model_weights.pth", map_location='cpu'))
model.eval()

onnx_model = PreprocessingWrapper(model)
onnx_model.eval()

dummy_input = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8)

torch.onnx.export(
    onnx_model,
    dummy_input,
    "classifier_with_preprocessing.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"}, "output": {0: "batch_size"}},
)
print("Exported to classifier_with_preprocessing.onnx")