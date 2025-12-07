"""
ONNX model parser to extract convolutional layer specifications.

Extracts layer dimensions and parameters needed for Timeloop problem specification.
"""
import onnx
from typing import Dict, List, Optional, Tuple
import numpy as np


class ConvLayerSpec:
    """Specification for a single convolutional layer."""
    
    def __init__(
        self,
        name: str,
        input_shape: Tuple[int, int, int, int],  # (N, C, H, W)
        output_shape: Tuple[int, int, int, int],  # (N, K, H_out, W_out)
        kernel_size: Tuple[int, int],  # (R, S)
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
    ):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Extract dimensions
        self.N = input_shape[0]  # Batch size
        self.C = input_shape[1]  # Input channels
        self.H = input_shape[2]  # Input height
        self.W = input_shape[3]  # Input width
        self.K = output_shape[1]  # Output channels
        self.R = kernel_size[0]   # Kernel height
        self.S = kernel_size[1]   # Kernel width
        self.P = output_shape[2]  # Output height
        self.Q = output_shape[3]  # Output width
    
    def __repr__(self):
        return (
            f"ConvLayerSpec(name={self.name}, "
            f"input=({self.N},{self.C},{self.H},{self.W}), "
            f"output=({self.N},{self.K},{self.P},{self.Q}), "
            f"kernel=({self.R},{self.S}))"
        )


def parse_onnx_model(model_path: str, layer_name: Optional[str] = None) -> List[ConvLayerSpec]:
    """
    Parse ONNX model and extract convolutional layer specifications.
    
    Args:
        model_path: Path to ONNX model file
        layer_name: Optional specific layer name to extract. If None, returns all conv layers.
    
    Returns:
        List of ConvLayerSpec objects
    """
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    
    # Build a graph to track tensor shapes
    shape_map = {}
    conv_layers = []
    
    # First pass: collect input shapes
    for input_tensor in model.graph.input:
        if input_tensor.type.tensor_type.shape.dim:
            shape = tuple(
                dim.dim_value if dim.dim_value > 0 else 1
                for dim in input_tensor.type.tensor_type.shape.dim
            )
            shape_map[input_tensor.name] = shape
    
    # Second pass: process nodes
    for node in model.graph.node:
        # Update shape map for outputs
        for output_name in node.output:
            if output_name not in shape_map:
                # Try to infer shape from input
                if node.input and node.input[0] in shape_map:
                    # Simple shape inference (can be extended)
                    input_shape = shape_map[node.input[0]]
                    if node.op_type == "Conv":
                        # This is simplified - real inference needs more logic
                        shape_map[output_name] = input_shape
                    else:
                        shape_map[output_name] = input_shape
        
        # Extract Conv layers
        if node.op_type == "Conv":
            if node.input and node.input[0] in shape_map:
                input_shape = shape_map[node.input[0]]
                
                # Get attributes
                kernel_shape = None
                strides = (1, 1)
                pads = (0, 0)
                group = 1
                
                for attr in node.attribute:
                    if attr.name == "kernel_shape":
                        kernel_shape = tuple(attr.ints)
                    elif attr.name == "strides":
                        strides = tuple(attr.ints)
                    elif attr.name == "pads":
                        pads = tuple(attr.ints)
                    elif attr.name == "group":
                        group = attr.i
                
                if kernel_shape is None:
                    # Try to infer from weight shape (if available)
                    continue
                
                # Calculate output shape (simplified)
                H, W = input_shape[2], input_shape[3]
                R, S = kernel_shape
                stride_h, stride_w = strides
                pad_h, pad_w = pads[:2] if len(pads) >= 2 else (0, 0)
                
                P = (H + 2 * pad_h - R) // stride_h + 1
                Q = (W + 2 * pad_w - S) // stride_w + 1
                
                # Get output channels from weight shape (if available in graph)
                K = input_shape[1]  # Default, should be inferred from weights
                
                output_shape = (input_shape[0], K, P, Q)
                
                spec = ConvLayerSpec(
                    name=node.name,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    kernel_size=kernel_shape,
                    stride=strides,
                    padding=(pad_h, pad_w),
                    groups=group,
                )
                
                if layer_name is None or node.name == layer_name:
                    conv_layers.append(spec)
    
    return conv_layers


def get_resnet18_conv3_example() -> ConvLayerSpec:
    """
    Create a synthetic example for ResNet-18 conv3 layer.
    This is used for testing when an actual ONNX model is not available.
    
    Typical ResNet-18 conv3 dimensions:
    - Input: (1, 64, 56, 56)  # After conv2
    - Output: (1, 128, 28, 28)  # After conv3
    - Kernel: 3x3
    - Stride: 2 (downsampling)
    """
    return ConvLayerSpec(
        name="conv3",
        input_shape=(1, 64, 56, 56),
        output_shape=(1, 128, 28, 28),
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
    )

