# Backbone Models

This directory contains implementations of various backbone architectures for deep learning tasks. Each backbone model is implemented as a standalone Python module and can be used interchangeably for different tasks.

---

## **Available Backbones**

1. **ResNet** (`resnet.py`): A customizable residual network architecture supporting `ResNet18`, `ResNet50`, and `ResNet101`.
2. **DLA** (`dla.py`): Deep Layer Aggregation (DLA) architecture for efficient feature reuse, supporting multiple configurations (`DLA34`, `DLA46`, `DLA60`, `DLA102`).
3. **RegNet** (`regnet.py`): A modern backbone for efficient neural network scaling, supporting `y` variants (`regnet-y-040`, `regnet-y-080`, etc.).

---

## **Input Requirements**

All backbone models take a **4D Tensor** as input:
- **Shape**: `(batch_size, channels, height, width)`
- **Expected Input**: RGB images normalized to a range of [0, 1] or [-1, 1], depending on the pretrained model.
- **Input Channel**: 3 (default for RGB images).

---

## **Output Details**

### **1. ResNet (`resnet.py`)**
- Supports `ResNet18`, `ResNet50`, and `ResNet101` by setting the `depth` parameter to `18`, `50`, or `101`.
- **Return Type**: List of feature maps from selected layers (controlled by `return_idx`).
- **Output Shapes**:
  - Each element in the output list corresponds to a specific layer's output.
  - Feature map resolutions are progressively reduced based on the ResNet stride (4, 8, 16, 32).
  - Example for input shape `(batch_size, 3, 224, 224)`:
    - `ResNet18`: `[torch.Size([batch_size, 64, 56, 56]), ..., torch.Size([batch_size, 512, 7, 7])]`
    - `ResNet50`: Similar to `ResNet18` but with larger feature dimensions due to `BottleNeck`.
    - `ResNet101`: Similar to `ResNet50` with deeper layers.

### **2. DLA (`dla.py`)**
- **Variants**: The DLA architecture has multiple configurations, such as `DLA34`, `DLA46`, `DLA60`, and `DLA102`. Each variant corresponds to a different network depth and complexity, with deeper networks capable of capturing more hierarchical features.
- **Return Type**: List of feature maps from selected levels (controlled by `return_index`).
- **Output Shapes**:
  - Each element in the output list corresponds to a DLA level.
  - Example for input shape `(batch_size, 3, 224, 224)`:
    - `[torch.Size([batch_size, 128, 56, 56]), ..., torch.Size([batch_size, 512, 7, 7])]`
  - Each variant will have a slightly different number of layers and feature map dimensions, but the output format remains consistent.

  Supported variants:
  - **DLA34**: Standard configuration with 34 layers.
  - **DLA46**: A variant with 46 layers.
  - **DLA60**: A variant with 60 layers.
  - **DLA102**: A deeper version with 102 layers.

### **3. RegNet (`regnet.py`)**
- **Variants**: The available variants for RegNet are all from the `y` series, such as `regnet-y-040`, `regnet-y-080`, `regnet-y-160`, etc. These variants differ in the number of channels and depth, providing a scalable architecture for efficient neural network design.
- **Return Type**: Selected hidden states from the RegNet model (controlled by `return_idx`).
- **Output Shapes**:
  - Each element corresponds to a hidden state from the backbone layers.
  - Example for input shape `(batch_size, 3, 224, 224)`:
    - `[torch.Size([batch_size, 128, 56, 56]), ..., torch.Size([batch_size, 1024, 7, 7])]`

---

## **Usage**

### **1. ResNet**
```python
import torch
from resnet import ResNet

# Initialize the ResNet model with depth=50 (ResNet50), return feature maps from layers 0, 1, and 2
model = ResNet(depth=50, pretrained=True, return_idx=[0, 1, 2])

input_tensor = torch.randn(8, 3, 224, 224)

output = model(input_tensor)

for idx, feature_map in enumerate(output):
    print(f"Feature Map {idx}: {feature_map.shape}")
```

### **2. DLA**
```python
import torch
from dla import DLANet

input = torch.randn(8, 3, 224, 224)

model = DLANet(dla='dla34', pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input = input.to(device)

model.eval()
with torch.no_grad(): 
    output = model(input)

print(f"Output shape: {output[0].shape}")
```

### **3. RegNet**
```python
import torch
from regnet import RegNet

model = RegNet(variant="regnet-y-080", return_idx=[0, 1, 2])

input_tensor = torch.randn(8, 3, 224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)

model.eval()
with torch.no_grad():
    output = model(input_tensor)

print(f"Output shapes: {[o.shape for o in output]}")
```

---

## **Testing Model Backbones**

- Navigate to your project directory:
  - `cd C:\VisionRD\AI-Hackathon24\arch\backbone`
  
- Create a testing script 'testing.py'
- Run the following command:

 ```python
python testing.py
```
---

## **Customization**

### **Pretrained Models**
- Models can be loaded with pretrained weights by setting `pretrained=True`.
- For RegNet, weights are loaded from the HuggingFace repository.

### **Selecting Features**
- The indices of the output layers can be customized using:
  - `return_idx` for ResNet and RegNet.
  - `return_index` for DLA.

---

## **Dependencies**
- Python >= 3.8
- PyTorch >= 1.7
- Transformers (for RegNet)

---

## **FAQ**

### Q1. How do I use these backbones in my own task?
- Import the desired backbone module.
- Pass an input tensor of shape `(batch_size, 3, height, width)`.
- Extract the output feature maps for downstream tasks like classification, segmentation, or detection.

### Q2. What are the advantages of each backbone?
- **ResNet**: Flexible and supports multiple ResNet depths (18, 50, 101).
- **DLA**: Efficient feature reuse with hierarchical structure, ideal for tasks requiring multi-scale feature extraction. Choose the appropriate variant depending on the required depth and complexity (DLA34, DLA46, DLA60, DLA102).
- **RegNet**: Scalable and efficient for modern architectures, with `y` variants offering a range of options for different computational budgets.
