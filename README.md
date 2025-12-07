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
│       ├── dataflow_validation.py    # IS/WS/RS template validation
│       └── batch_optimizer.py        # Batch optimization utilities
├── examples/
│   ├── test_single_gene.py          # Step 2: Single gene evaluation
│   ├── validate_dataflows.py        # Step 3: Dataflow validation
│   ├── run_optimization.py          # Step 5: NSGA-II optimization
│   ├── use_real_model.py            # Using real ONNX models
│   └── batch_optimize_all_models.py # Batch optimization for all models
├── models/                           # ONNX model files (place your .onnx files here)
│   └── README.md                     # Instructions for getting models
├── timeloop_output/                  # Timeloop execution outputs
│   └── README.md                     # Information about Timeloop outputs
├── scripts/
│   └── download_model.py            # Helper script to download/convert models
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

**1. Setup the environment:**
```bash
uv sync
```

**2. Run the main demonstration:**
```bash
uv run python main.py
```

This demonstrates the complete pipeline:
1. Creating example genes
2. Loading layer specifications
3. Evaluating genes with Timeloop
4. Validating dataflow templates
5. Testing fitness functions
6. Running NSGA-II optimization (if enabled)

**3. Run NSGA-II optimization:**
```bash
uv run python examples/run_optimization.py
```

### Run NSGA-II Optimization

**Simplest way** (uses synthetic example):
```bash
uv run python examples/run_optimization.py
```

**With real model**:
```bash
# Download a model first
uv run python scripts/download_model.py resnet18

# Then optimize
uv run python examples/use_real_model.py
# (Uncomment the optimization code in the script)
```

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

#### Using Real ONNX Models

```bash
# First, download a model
uv run python scripts/download_model.py resnet18

# Then use it
uv run python examples/use_real_model.py
```

This example shows how to:
- Load a real ONNX model from `models/` directory
- Extract convolutional layers
- Optimize real CNN layers

#### Batch Optimization (All Models)

```bash
# Optimize all models in models/ directory
uv run python examples/batch_optimize_all_models.py
```

This will:
- Find all `.onnx` files in `models/`
- Optimize all convolutional layers from each model
- Generate summary statistics

**Note**: This can take a very long time! Adjust `max_layers_per_model` in the script to limit layers per model.

### Getting Models

Download or convert models to ONNX format:

```bash
# Using the helper script (requires PyTorch)
uv run python scripts/download_model.py resnet18
uv run python scripts/download_model.py resnet50
uv run python scripts/download_model.py mobilenet_v2

# Or manually convert from PyTorch/TensorFlow
# See models/README.md for details
```

### Programmatic Usage

```python
from pe_mapper import (
    Gene,
    get_resnet18_conv3_example,
    parse_onnx_model,
    evaluate_gene,
    optimize_genes,
)

# Option 1: Use synthetic example
layer_spec = get_resnet18_conv3_example()

# Option 2: Parse from ONNX model
from pe_mapper import parse_onnx_model
layers = parse_onnx_model("models/resnet18.onnx")
layer_spec = layers[0]  # Use first conv layer

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

# Run optimization (single layer)
pareto_genes, objectives = optimize_genes(
    layer_spec,
    population_size=20,
    n_generations=10,
)

# Batch optimization (all models in models/ directory)
from pe_mapper import optimize_all_models, summarize_results

results = optimize_all_models(
    models_dir="models",
    population_size=20,
    n_generations=10,
    max_layers_per_model=3,  # Optional: limit layers per model
)
summarize_results(results)
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

## Batch Optimization

The project supports batch processing to optimize multiple models and layers:

### Optimize All Models
```python
from pe_mapper import optimize_all_models, summarize_results

# Optimize all .onnx files in models/ directory
results = optimize_all_models(
    models_dir="models",
    population_size=20,
    n_generations=10,
    max_layers_per_model=3,  # Optional: limit for faster testing
)

# Print summary
summarize_results(results)
```

### Optimize All Layers from One Model
```python
from pe_mapper import optimize_all_layers_from_model

# Optimize all conv layers from a single model
results = optimize_all_layers_from_model(
    "models/resnet18.onnx",
    population_size=20,
    n_generations=10,
)
```

### Command Line
```bash
# Optimize all models in models/ directory
uv run python examples/batch_optimize_all_models.py
```

**Note**: Batch optimization can take a very long time. Adjust `max_layers_per_model` to limit the number of layers processed per model.

## Development Plan

This project follows a 5-step development plan:

1. ✅ **Fix scope and gene definition**: Minimal but complete gene for one conv layer
2. ✅ **Build evaluation backbone**: ONNX → Timeloop → metrics pipeline
3. ✅ **Validate dataflow templates**: Handcrafted IS/WS/RS mappings
4. ✅ **Wrap as fitness function**: Stateless, deterministic evaluation
5. ✅ **Plug in NSGA-II**: Multi-objective optimization with pymoo

## Notes

- **Mock Metrics**: If Timeloop is not installed, the system uses heuristic-based mock metrics for testing. This allows you to test the full pipeline without Timeloop.
- **Timeloop Integration**: The Timeloop YAML generation is simplified; production use may require more detailed specs. The code structure is ready for real Timeloop integration.
- **Gene Encoding**: The current gene encoding is minimal but complete; can be extended for more complex scenarios.
- **Constraints**: Genes are automatically validated and repaired to satisfy layer dimension constraints.
- **Batch Processing**: The batch optimizer processes models sequentially. For very large batches, consider parallel processing or limiting layers per model.
- **Performance**: NSGA-II runtime depends on population size, generations, and Timeloop evaluation time. Start with small parameters for testing.

## Features

✅ **Gene-based encoding**: Complete hardware and mapping configuration  
✅ **Multiple dataflows**: IS, WS, and RS templates  
✅ **ONNX integration**: Parse real CNN models  
✅ **Timeloop integration**: Generate architecture and mapping specs  
✅ **NSGA-II optimization**: Multi-objective Pareto-optimal solutions  
✅ **Batch processing**: Optimize multiple models and layers automatically  
✅ **Mock metrics**: Works without Timeloop for testing  

## Future Work

- [ ] More sophisticated Timeloop mapping generation
- [ ] Additional dataflow patterns
- [ ] Integration with actual hardware measurements
- [ ] Visualization of Pareto fronts
- [ ] Support for other layer types (depthwise, grouped convolutions)
- [ ] Result persistence (save/load optimization results)

## License

This project is part of CS454 coursework.

## Authors

CS454 Team 14
