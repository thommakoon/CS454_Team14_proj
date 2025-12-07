"""
Quick verification script to test if the environment is set up correctly.
"""
import sys

def check_imports():
    """Check if all required packages can be imported."""
    print("Checking imports...")
    
    try:
        import numpy
        print(f"  ✓ numpy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import onnx
        print(f"  ✓ onnx {onnx.__version__}")
    except ImportError as e:
        print(f"  ✗ onnx: {e}")
        return False
    
    try:
        import pymoo
        print(f"  ✓ pymoo {pymoo.__version__}")
    except ImportError as e:
        print(f"  ✗ pymoo: {e}")
        return False
    
    try:
        import yaml
        print(f"  ✓ pyyaml")
    except ImportError as e:
        print(f"  ✗ pyyaml: {e}")
        return False
    
    return True

def check_project_imports():
    """Check if the project package can be imported."""
    print("\nChecking project package...")
    
    try:
        from pe_mapper import Gene, get_resnet18_conv3_example
        print("  ✓ pe_mapper package imported successfully")
        
        # Test creating a gene
        gene = Gene(
            pe_array_x=4,
            pe_array_y=4,
            simd_width=4,
            buffer_budget_class=1,
            dataflow="IS",
            tile_n=1,
            tile_c=16,
            tile_k=32,
            tile_h=7,
            tile_w=7,
            spatial_n=False,
            spatial_c=True,
            spatial_k=False,
            spatial_h=True,
            spatial_w=True,
            loop_permutation_id=0,
        )
        print(f"  ✓ Gene creation works: {gene.dataflow} dataflow")
        
        # Test layer spec
        layer_spec = get_resnet18_conv3_example()
        print(f"  ✓ Layer spec works: {layer_spec.name}")
        
        return True
    except Exception as e:
        print(f"  ✗ pe_mapper import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Environment Verification")
    print("=" * 60)
    
    print(f"\nPython version: {sys.version}")
    print(f"Python path: {sys.executable}")
    
    all_ok = True
    
    # Check dependencies
    if not check_imports():
        all_ok = False
    
    # Check project package
    if not check_project_imports():
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All checks passed! Environment is set up correctly.")
        print("=" * 60)
        return 0
    else:
        print("✗ Some checks failed. Please run 'uv sync' to install dependencies.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())

