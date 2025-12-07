"""
Validation script for IS/WS/RS dataflow templates.

This script implements Step 3 of the plan: validating dataflow templates
before using them in NSGA-II optimization.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pe_mapper import get_resnet18_conv3_example
from pe_mapper.dataflow_validation import (
    validate_dataflow_templates,
    create_handcrafted_is_gene,
    create_handcrafted_ws_gene,
    create_handcrafted_rs_gene,
)


def main():
    """Validate all three dataflow templates."""
    print("=" * 60)
    print("Dataflow Template Validation")
    print("=" * 60)
    
    # Get layer spec
    layer_spec = get_resnet18_conv3_example()
    print(f"\nLayer: {layer_spec.name}")
    print(f"  Input: {layer_spec.input_shape}")
    print(f"  Output: {layer_spec.output_shape}")
    print(f"  Kernel: {layer_spec.kernel_size}")
    
    # Create handcrafted genes for each dataflow
    print("\nCreating handcrafted genes for each dataflow...")
    is_gene = create_handcrafted_is_gene(layer_spec)
    ws_gene = create_handcrafted_ws_gene(layer_spec)
    rs_gene = create_handcrafted_rs_gene(layer_spec)
    
    print(f"  IS gene: PE array {is_gene.pe_array_x}x{is_gene.pe_array_y}, "
          f"spatial C/H/W")
    print(f"  WS gene: PE array {ws_gene.pe_array_x}x{ws_gene.pe_array_y}, "
          f"spatial K/H/W")
    print(f"  RS gene: PE array {rs_gene.pe_array_x}x{rs_gene.pe_array_y}, "
          f"spatial C/K/H/W")
    
    # Validate templates
    print("\n" + "=" * 60)
    results = validate_dataflow_templates(layer_spec)
    
    # Detailed analysis
    print("\n" + "=" * 60)
    print("Expected Reuse Patterns:")
    print("=" * 60)
    print("\nInput-Stationary (IS):")
    print("  - Inputs: Stationary in PE buffers (low reads)")
    print("  - Weights: Move through PEs (higher movement)")
    print("  - Outputs: Accumulate and move (higher movement)")
    
    print("\nWeight-Stationary (WS):")
    print("  - Weights: Stationary in PE buffers (low reads)")
    print("  - Inputs: Move through PEs (higher movement)")
    print("  - Outputs: Accumulate and move (higher movement)")
    
    print("\nRow-Stationary (RS):")
    print("  - Row of weights: Stationary")
    print("  - Row of inputs: Stationary")
    print("  - Outputs: Accumulate")
    print("  - More balanced reuse of all three")
    
    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)
    print("\nNote: If Timeloop is not installed, mock metrics are used.")
    print("      Install Timeloop for accurate buffer traffic analysis.")


if __name__ == "__main__":
    main()

