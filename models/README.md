# Models Directory

Place your ONNX model files here.

## Getting Models

### Option 1: Download Pre-trained Models
- PyTorch Hub: https://pytorch.org/hub/
- ONNX Model Zoo: https://github.com/onnx/models

### Option 2: Convert from PyTorch
```python
import torch
import torchvision.models as models

# Load a model
model = models.resnet18(pretrained=True)
model.eval()

# Convert to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "models/resnet18.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

### Option 3: Convert from TensorFlow
```python
import tensorflow as tf
import tf2onnx

# Load model and convert
# See tf2onnx documentation
```

## Example Models to Try
- ResNet-18, ResNet-34, ResNet-50
- VGG-16, VGG-19
- MobileNet-V2, MobileNet-V3
- EfficientNet variants

## Usage
```python
from pe_mapper import parse_onnx_model

# Parse a model
layers = parse_onnx_model("models/resnet18.onnx")
print(f"Found {len(layers)} convolutional layers")
```

