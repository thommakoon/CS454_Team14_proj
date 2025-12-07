# CS454 Team 14: PE-focused CNN Accelerator Mapping Optimization

A GenCNN-like approach for optimizing convolutional layer mappings on Processing Element (PE) arrays using Timeloop, ONNX, and NSGA-II.

## Overview

This project implements a complete pipeline for exploring the design space of CNN accelerator mappings:

1. **Gene Definition**: Encodes hardware configuration (PE array, SIMD, buffers) and mapping strategy (tiling, partitioning, dataflow)
2. **ONNX Integration**: Extracts convolutional layer specifications from ONNX models
3. **Timeloop Integration**: Generates architecture and mapping specs, runs Timeloop for accurate performance/energy evaluation
4. **Dataflow Templates**: Implements Input-Stationary (IS), Weight-Stationary (WS), and Row-Stationary (RS) dataflows
5. **NSGA-II Optimization**: Multi-objective evolutionary algorithm to find Pareto-optimal mappings

## Project Structure

```
CS454_Team14_proj/
├── src/
│   └── pe_mapper/
│       ├── __init__.py              # Package exports
│       ├── gene.py                   # Gene definition and encoding
│       ├── onnx_parser.py            # ONNX model parsing
│       ├── timeloop_integration.py   # Timeloop YAML generation and execution
│       ├── fitness.py                # Fitness evaluation function
│       ├── nsga2_optimizer.py        # NSGA-II optimization
│       └── dataflow_validation.py    # IS/WS/RS template validation
├── examples/
│   ├── test_single_gene.py          # Step 2: Single gene evaluation
│   ├── validate_dataflows.py        # Step 3: Dataflow validation
│   └── run_optimization.py          # Step 5: NSGA-II optimization
├── main.py                           # Main entry point with full pipeline demo
├── pyproject.toml                    # Project dependencies
└── README.md                         # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher (or use `uv` to manage Python versions)
- (Optional) Timeloop mapper for accurate evaluation

### Install with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

**Windows:**
```cmd
# Install uv if you haven't already
# See: https://github.com/astral-sh/uv

# Setup the project (creates venv and installs dependencies)
setup_uv.bat

# Or manually:
uv sync
```

**Linux/Mac:**
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup the project
chmod +x setup_uv.sh
./setup_uv.sh

# Or manually:
uv sync
```

**Run the project:**
```bash
# Using uv run (recommended)
uv run python main.py

# Or activate the virtual environment
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
python main.py
```

### Install with pip (Alternative)

```bash
pip install -e .
```

### Dependencies

This will install:
- `numpy`: Numerical computations
- `onnx`: ONNX model parsing
- `pymoo`: NSGA-II optimization
- `pyyaml`: YAML file generation for Timeloop

### Optional: Install Timeloop

For accurate performance evaluation, install Timeloop:
```bash
# Follow Timeloop installation instructions at:
# https://github.com/NVlabs/timeloop
```

If Timeloop is not available, the system will use mock metrics for testing.

## Usage

### Quick Start

Run the main demonstration:

```bash
python main.py
```

This demonstrates the complete pipeline:
1. Creating example genes
2. Loading layer specifications
3. Evaluating genes with Timeloop
4. Validating dataflow templates
5. Testing fitness functions

### Step-by-Step Examples

#### Step 2: Evaluate a Single Gene

```bash
python examples/test_single_gene.py
```

Tests the evaluation backbone by:
- Creating a fixed gene instance
- Generating Timeloop specs
- Running Timeloop (or mock)
- Verifying metrics change with different dataflows

#### Step 3: Validate Dataflow Templates

```bash
python examples/validate_dataflows.py
```

Validates IS/WS/RS dataflow implementations:
- Creates handcrafted genes for each dataflow
- Verifies expected reuse patterns
- Compares metrics across dataflows

#### Step 5: Run NSGA-II Optimization

```bash
python examples/run_optimization.py
```

Runs multi-objective optimization:
- Population size: 20
- Generations: 10
- Objectives: (latency, energy, -utilization)
- Returns Pareto-optimal solutions

### Programmatic Usage

```python
from pe_mapper import (
    Gene,
    get_resnet18_conv3_example,
    evaluate_gene,
    optimize_genes,
)

# Get layer specification
layer_spec = get_resnet18_conv3_example()

# Create a gene
gene = Gene(
    pe_array_x=8,
    pe_array_y=8,
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

# Evaluate gene
latency, energy, neg_utilization = evaluate_gene(gene, layer_spec)
utilization = -neg_utilization

# Run optimization
pareto_genes, objectives = optimize_genes(
    layer_spec,
    population_size=20,
    n_generations=10,
)
```

## Gene Definition

A `Gene` encodes a complete mapping strategy:

### Hardware Configuration
- `pe_array_x`, `pe_array_y`: PE array dimensions
- `simd_width`: SIMD width per PE
- `buffer_budget_class`: Buffer size class (0=small, 1=medium, 2=large)
- `dataflow`: Dataflow mode ("IS", "WS", or "RS")

### Mapping Configuration
- `tile_n`, `tile_c`, `tile_k`, `tile_h`, `tile_w`: Tiling factors
- `spatial_*`: Which dimensions are spatially mapped vs temporally iterated
- `loop_permutation_id`: Execution order (0-119 for 5! permutations)

## Dataflow Modes

### Input-Stationary (IS)
- **Stationary**: Input activations stay in PE buffers
- **Moving**: Weights and outputs move through PEs
- **Best for**: High input reuse scenarios

### Weight-Stationary (WS)
- **Stationary**: Weights stay in PE buffers
- **Moving**: Inputs and outputs move through PEs
- **Best for**: High weight reuse scenarios

### Row-Stationary (RS)
- **Stationary**: Row of weights and inputs
- **Moving**: Outputs accumulate
- **Best for**: Balanced reuse of all three data types

## Objectives

The optimization minimizes three objectives:
1. **Latency**: Execution time (seconds)
2. **Energy**: Total energy consumption (pJ)
3. **-Utilization**: Negative PE utilization (to maximize utilization)

NSGA-II finds the Pareto front of solutions that trade off these objectives.

## Development Plan

This project follows a 5-step development plan:

1. ✅ **Fix scope and gene definition**: Minimal but complete gene for one conv layer
2. ✅ **Build evaluation backbone**: ONNX → Timeloop → metrics pipeline
3. ✅ **Validate dataflow templates**: Handcrafted IS/WS/RS mappings
4. ✅ **Wrap as fitness function**: Stateless, deterministic evaluation
5. ✅ **Plug in NSGA-II**: Multi-objective optimization with pymoo

## Notes

- **Mock Metrics**: If Timeloop is not installed, the system uses heuristic-based mock metrics for testing
- **Timeloop Integration**: The Timeloop YAML generation is simplified; production use may require more detailed specs
- **Gene Encoding**: The current gene encoding is minimal; can be extended for more complex scenarios
- **Constraints**: Genes are automatically validated and repaired to satisfy layer dimension constraints

## Future Work

- [ ] Support for multiple layers
- [ ] More sophisticated Timeloop mapping generation
- [ ] Additional dataflow patterns
- [ ] Integration with actual hardware measurements
- [ ] Visualization of Pareto fronts
- [ ] Support for other layer types (depthwise, grouped convolutions)

## License

This project is part of CS454 coursework.

## Authors

CS454 Team 14
