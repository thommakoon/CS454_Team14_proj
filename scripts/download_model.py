"""
Helper script to download and convert models to ONNX format.

Examples:
    python scripts/download_model.py resnet18
    python scripts/download_model.py resnet50 --output models/
    python scripts/download_model.py mobilenet_v2
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def download_resnet(model_name: str, output_dir: str = "models"):
    """Download ResNet model and convert to ONNX."""
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("Error: PyTorch not installed. Install with: pip install torch torchvision")
        return False
    
    print(f"Downloading {model_name}...")
    
    # Map model names to functions
    model_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }
    
    if model_name not in model_map:
        print(f"Error: Unknown model {model_name}")
        print(f"Available: {list(model_map.keys())}")
        return False
    
    # Load model
    model = model_map[model_name](pretrained=True)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Convert to ONNX
    output_path = Path(output_dir) / f"{model_name}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11,
    )
    
    print(f"✓ Successfully saved to {output_path}")
    return True


def download_mobilenet(model_name: str, output_dir: str = "models"):
    """Download MobileNet model and convert to ONNX."""
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("Error: PyTorch not installed. Install with: pip install torch torchvision")
        return False
    
    print(f"Downloading {model_name}...")
    
    model_map = {
        "mobilenet_v2": models.mobilenet_v2,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "mobilenet_v3_small": models.mobilenet_v3_small,
    }
    
    if model_name not in model_map:
        print(f"Error: Unknown model {model_name}")
        print(f"Available: {list(model_map.keys())}")
        return False
    
    model = model_map[model_name](pretrained=True)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = Path(output_dir) / f"{model_name}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11,
    )
    
    print(f"✓ Successfully saved to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and convert models to ONNX")
    parser.add_argument("model", help="Model name (e.g., resnet18, mobilenet_v2)")
    parser.add_argument("--output", "-o", default="models", help="Output directory")
    
    args = parser.parse_args()
    
    model_name = args.model.lower()
    
    # Try ResNet first
    if model_name.startswith("resnet"):
        success = download_resnet(model_name, args.output)
    elif model_name.startswith("mobilenet"):
        success = download_mobilenet(model_name, args.output)
    else:
        print(f"Error: Unknown model type: {model_name}")
        print("Supported: resnet18, resnet34, resnet50, mobilenet_v2, etc.")
        success = False
    
    if success:
        print("\nNext steps:")
        print(f"  from pe_mapper import parse_onnx_model")
        print(f"  layers = parse_onnx_model('{args.output}/{model_name}.onnx')")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

