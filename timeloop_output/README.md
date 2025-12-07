# Timeloop Output Directory

This directory stores Timeloop execution outputs and generated YAML files.

## Structure

When you run Timeloop through the `TimeloopRunner`, it will:
1. Generate YAML files (arch.yaml, problem.yaml, mapping.yaml)
2. Execute Timeloop mapper
3. Store output files including:
   - `timeloop-mapper.stats.txt` - Performance statistics
   - `timeloop-mapper.map.yaml` - Generated mapping
   - Other Timeloop output files

## Temporary vs Persistent Outputs

- **Temporary outputs**: By default, `TimeloopRunner` uses temporary directories
- **Persistent outputs**: To keep outputs, specify `output_dir` parameter:
  ```python
  runner = TimeloopRunner()
  metrics = runner.run_timeloop(gene, layer_spec, output_dir="timeloop_output/layer1")
  ```

## Files Generated

For each evaluation, you'll typically see:
- `arch.yaml` - Architecture specification
- `problem.yaml` - Problem (layer) specification  
- `mapping.yaml` - Mapping specification
- `timeloop-mapper.stats.txt` - Metrics (latency, energy, utilization, etc.)
- `timeloop-mapper.map.yaml` - Final mapping solution

## Cleaning Up

To clean up old outputs:
```bash
# Remove all outputs
rm -rf timeloop_output/*

# Or keep specific runs
rm -rf timeloop_output/layer1
```

