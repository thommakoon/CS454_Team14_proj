"""
Timeloop integration for generating architecture and mapping specifications.

Converts gene and layer spec into Timeloop YAML files and executes Timeloop.
"""
import yaml
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

from .gene import Gene
from .onnx_parser import ConvLayerSpec


class TimeloopRunner:
    """Handles Timeloop execution and result parsing."""
    
    def __init__(self, timeloop_path: Optional[str] = None):
        """
        Initialize Timeloop runner.
        
        Args:
            timeloop_path: Path to Timeloop executable. If None, assumes 'timeloop-mapper' is in PATH.
        """
        self.timeloop_path = timeloop_path or "timeloop-mapper"
        self.temp_dir = None
    
    def generate_architecture_spec(self, gene: Gene) -> Dict[str, Any]:
        """
        Generate Timeloop architecture specification from gene.
        
        Returns:
            Dictionary representing architecture YAML structure
        """
        input_buf, weight_buf, output_buf = gene.get_buffer_sizes()
        
        arch = {
            "architecture": {
                "version": "0.3",
                "name": "PE_Array",
                "targets": {
                    "target": {
                        "name": "PE_Array",
                        "type": "pe_array",
                        "instances": [
                            {
                                "name": "PE",
                                "count": gene.pe_array_x * gene.pe_array_y,
                                "attributes": {
                                    "width": gene.pe_array_x,
                                    "height": gene.pe_array_y,
                                }
                            }
                        ],
                        "subtree": [
                            {
                                "name": "GlobalBuffer",
                                "type": "buffer",
                                "attributes": {
                                    "depth": max(input_buf, weight_buf, output_buf) // 4,  # Assume 4-byte words
                                    "width": 64,  # 64-bit words
                                    "bandwidth": 64,
                                },
                                "subtree": [
                                    {
                                        "name": "PEArray",
                                        "type": "pe_array",
                                        "attributes": {
                                            "width": gene.pe_array_x,
                                            "height": gene.pe_array_y,
                                        },
                                        "subtree": [
                                            {
                                                "name": "PE",
                                                "type": "pe",
                                                "attributes": {
                                                    "vector_width": gene.simd_width,
                                                },
                                                "subtree": [
                                                    {
                                                        "name": "InputBuffer",
                                                        "type": "buffer",
                                                        "attributes": {
                                                            "depth": input_buf // 4,
                                                            "width": 32,
                                                            "bandwidth": 32,
                                                        }
                                                    },
                                                    {
                                                        "name": "WeightBuffer",
                                                        "type": "buffer",
                                                        "attributes": {
                                                            "depth": weight_buf // 4,
                                                            "width": 32,
                                                            "bandwidth": 32,
                                                        }
                                                    },
                                                    {
                                                        "name": "OutputBuffer",
                                                        "type": "buffer",
                                                        "attributes": {
                                                            "depth": output_buf // 4,
                                                            "width": 32,
                                                            "bandwidth": 32,
                                                        }
                                                    },
                                                    {
                                                        "name": "MAC",
                                                        "type": "compute",
                                                        "attributes": {
                                                            "op_type": "mul_add",
                                                        }
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        }
        return arch
    
    def generate_problem_spec(self, layer_spec: ConvLayerSpec) -> Dict[str, Any]:
        """
        Generate Timeloop problem specification from layer spec.
        
        Returns:
            Dictionary representing problem YAML structure
        """
        problem = {
            "problem": {
                "version": "0.3",
                "shape": {
                    "N": layer_spec.N,
                    "C": layer_spec.C,
                    "K": layer_spec.K,
                    "H": layer_spec.H,
                    "W": layer_spec.W,
                    "R": layer_spec.R,
                    "S": layer_spec.S,
                    "P": layer_spec.P,
                    "Q": layer_spec.Q,
                },
                "densities": {
                    "Inputs": 1.0,
                    "Weights": 1.0,
                    "Outputs": 1.0,
                }
            }
        }
        return problem
    
    def generate_mapping_spec(self, gene: Gene, layer_spec: ConvLayerSpec) -> Dict[str, Any]:
        """
        Generate Timeloop mapping specification from gene.
        
        Implements IS, WS, and RS dataflow templates.
        
        Returns:
            Dictionary representing mapping YAML structure
        """
        # Get loop order
        loop_order = gene.get_loop_order()
        
        # Build mapping based on dataflow
        if gene.dataflow == "IS":
            # Input-Stationary: inputs stay in PE buffers, weights/outputs move
            mapping = self._generate_is_mapping(gene, layer_spec, loop_order)
        elif gene.dataflow == "WS":
            # Weight-Stationary: weights stay in PE buffers, inputs/outputs move
            mapping = self._generate_ws_mapping(gene, layer_spec, loop_order)
        elif gene.dataflow == "RS":
            # Row-Stationary: row of weights and inputs stay, outputs accumulate
            mapping = self._generate_rs_mapping(gene, layer_spec, loop_order)
        else:
            raise ValueError(f"Unknown dataflow: {gene.dataflow}")
        
        return mapping
    
    def _generate_is_mapping(
        self, gene: Gene, layer_spec: ConvLayerSpec, loop_order: List[str]
    ) -> Dict[str, Any]:
        """Generate Input-Stationary mapping."""
        # IS: Inputs are stationary in PE buffers
        # Map spatial dimensions (H, W) across PE array
        # Temporal iteration over K, C, R, S
        
        mapping = {
            "version": "0.3",
            "target": "PE_Array",
            "bypass": [],
            "nest": []
        }
        
        # Determine spatial vs temporal mapping
        spatial_dims = []
        temporal_dims = []
        
        dim_map = {
            "N": ("N", gene.tile_n, gene.spatial_n),
            "C": ("C", gene.tile_c, gene.spatial_c),
            "K": ("K", gene.tile_k, gene.spatial_k),
            "H": ("H", gene.tile_h, gene.spatial_h),
            "W": ("W", gene.tile_w, gene.spatial_w),
        }
        
        for dim_name in loop_order:
            dim_key, tile_size, is_spatial = dim_map[dim_name]
            if is_spatial:
                spatial_dims.append((dim_key, tile_size))
            else:
                temporal_dims.append((dim_key, tile_size))
        
        # Build nest structure (simplified - real Timeloop mapping is more complex)
        nest = []
        
        # Global buffer level
        for dim_key, tile_size in temporal_dims:
            nest.append({
                "target": "GlobalBuffer",
                "type": "temporal",
                "factors": f"{dim_key}={tile_size}",
            })
        
        # PE array level - spatial mapping
        for dim_key, tile_size in spatial_dims[:2]:  # Map up to 2 dims spatially
            nest.append({
                "target": "PEArray",
                "type": "spatial",
                "factors": f"{dim_key}={tile_size}",
            })
        
        # PE level - keep inputs stationary
        nest.append({
            "target": "PE",
            "type": "temporal",
            "factors": "Inputs=1",  # Keep inputs stationary
        })
        
        mapping["nest"] = nest
        return mapping
    
    def _generate_ws_mapping(
        self, gene: Gene, layer_spec: ConvLayerSpec, loop_order: List[str]
    ) -> Dict[str, Any]:
        """Generate Weight-Stationary mapping."""
        # WS: Weights are stationary in PE buffers
        # Similar structure but keep weights stationary
        
        mapping = {
            "version": "0.3",
            "target": "PE_Array",
            "bypass": [],
            "nest": []
        }
        
        dim_map = {
            "N": ("N", gene.tile_n, gene.spatial_n),
            "C": ("C", gene.tile_c, gene.spatial_c),
            "K": ("K", gene.tile_k, gene.spatial_k),
            "H": ("H", gene.tile_h, gene.spatial_h),
            "W": ("W", gene.tile_w, gene.spatial_w),
        }
        
        spatial_dims = []
        temporal_dims = []
        
        for dim_name in loop_order:
            dim_key, tile_size, is_spatial = dim_map[dim_name]
            if is_spatial:
                spatial_dims.append((dim_key, tile_size))
            else:
                temporal_dims.append((dim_key, tile_size))
        
        nest = []
        
        for dim_key, tile_size in temporal_dims:
            nest.append({
                "target": "GlobalBuffer",
                "type": "temporal",
                "factors": f"{dim_key}={tile_size}",
            })
        
        for dim_key, tile_size in spatial_dims[:2]:
            nest.append({
                "target": "PEArray",
                "type": "spatial",
                "factors": f"{dim_key}={tile_size}",
            })
        
        nest.append({
            "target": "PE",
            "type": "temporal",
            "factors": "Weights=1",  # Keep weights stationary
        })
        
        mapping["nest"] = nest
        return mapping
    
    def _generate_rs_mapping(
        self, gene: Gene, layer_spec: ConvLayerSpec, loop_order: List[str]
    ) -> Dict[str, Any]:
        """Generate Row-Stationary mapping."""
        # RS: Row of weights and inputs stay, outputs accumulate
        # More balanced reuse pattern
        
        mapping = {
            "version": "0.3",
            "target": "PE_Array",
            "bypass": [],
            "nest": []
        }
        
        dim_map = {
            "N": ("N", gene.tile_n, gene.spatial_n),
            "C": ("C", gene.tile_c, gene.spatial_c),
            "K": ("K", gene.tile_k, gene.spatial_k),
            "H": ("H", gene.tile_h, gene.spatial_h),
            "W": ("W", gene.tile_w, gene.spatial_w),
        }
        
        spatial_dims = []
        temporal_dims = []
        
        for dim_name in loop_order:
            dim_key, tile_size, is_spatial = dim_map[dim_name]
            if is_spatial:
                spatial_dims.append((dim_key, tile_size))
            else:
                temporal_dims.append((dim_key, tile_size))
        
        nest = []
        
        for dim_key, tile_size in temporal_dims:
            nest.append({
                "target": "GlobalBuffer",
                "type": "temporal",
                "factors": f"{dim_key}={tile_size}",
            })
        
        for dim_key, tile_size in spatial_dims[:2]:
            nest.append({
                "target": "PEArray",
                "type": "spatial",
                "factors": f"{dim_key}={tile_size}",
            })
        
        # RS: keep row of inputs and weights
        nest.append({
            "target": "PE",
            "type": "temporal",
            "factors": "Inputs=1,Weights=1",  # Keep both stationary
        })
        
        mapping["nest"] = nest
        return mapping
    
    def run_timeloop(
        self,
        gene: Gene,
        layer_spec: ConvLayerSpec,
        output_dir: Optional[str] = None,
        keep_outputs: bool = False
    ) -> Dict[str, float]:
        """
        Run Timeloop with given gene and layer spec.
        
        Args:
            gene: The gene to evaluate
            layer_spec: The layer specification
            output_dir: Optional directory to save outputs. If None, uses temp directory.
            keep_outputs: If True and output_dir is None, saves to timeloop_output/ with layer name.
        
        Returns:
            Dictionary with metrics: latency, energy, utilization, etc.
        """
        if output_dir is None:
            if keep_outputs:
                # Save to timeloop_output directory
                output_dir = os.path.join("timeloop_output", f"{layer_spec.name}_{gene.dataflow}")
                os.makedirs(output_dir, exist_ok=True)
            else:
                # Use temporary directory
                self.temp_dir = tempfile.mkdtemp()
                output_dir = self.temp_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate specs
        arch_spec = self.generate_architecture_spec(gene)
        problem_spec = self.generate_problem_spec(layer_spec)
        mapping_spec = self.generate_mapping_spec(gene, layer_spec)
        
        # Write YAML files
        arch_path = os.path.join(output_dir, "arch.yaml")
        problem_path = os.path.join(output_dir, "problem.yaml")
        mapping_path = os.path.join(output_dir, "mapping.yaml")
        
        with open(arch_path, 'w') as f:
            yaml.dump(arch_spec, f, default_flow_style=False)
        
        with open(problem_path, 'w') as f:
            yaml.dump(problem_spec, f, default_flow_style=False)
        
        with open(mapping_path, 'w') as f:
            yaml.dump(mapping_spec, f, default_flow_style=False)
        
        # Run Timeloop (if available)
        # For now, return mock metrics if Timeloop is not available
        try:
            result = subprocess.run(
                [self.timeloop_path, arch_path, problem_path, mapping_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                # Parse Timeloop output (simplified - real parsing needs more work)
                metrics = self._parse_timeloop_output(result.stdout, output_dir)
            else:
                # Fallback to mock metrics for testing
                metrics = self._generate_mock_metrics(gene, layer_spec)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Timeloop not available, use mock metrics
            # print("Timeloop not available, using mock metrics")
            metrics = self._generate_mock_metrics(gene, layer_spec)
        
        return metrics
    
    def _parse_timeloop_output(self, stdout: str, output_dir: str) -> Dict[str, float]:
        """
        Parse Timeloop output to extract metrics.
        
        This is a simplified parser - real implementation would need to
        parse Timeloop's detailed output format.
        """
        metrics = {
            "latency": 0.0,
            "energy": 0.0,
            "utilization": 0.0,
            "area": 0.0,
        }
        
        # Try to read stats file if it exists
        stats_path = os.path.join(output_dir, "timeloop-mapper.stats.txt")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                content = f.read()
                # Simple parsing (would need more robust parsing in production)
                # This is a placeholder
                pass
        
        return metrics
    
    def _generate_mock_metrics(
        self, gene: Gene, layer_spec: ConvLayerSpec
    ) -> Dict[str, float]:
        """
        Generate mock metrics for testing when Timeloop is not available.
        
        These metrics are heuristic-based and should be replaced with
        actual Timeloop results in production.
        """
        # Heuristic calculations
        total_ops = (
            layer_spec.N * layer_spec.K * layer_spec.P * layer_spec.Q *
            layer_spec.C * layer_spec.R * layer_spec.S
        )
        
        pe_count = gene.pe_array_x * gene.pe_array_y
        
        # Latency: operations / (PEs * SIMD * frequency estimate)
        # Assume 1 GHz frequency
        frequency = 1e9
        cycles = total_ops / (pe_count * gene.simd_width)
        latency = cycles / frequency  # seconds
        
        # Energy: rough estimate based on buffer sizes and operations
        input_buf, weight_buf, output_buf = gene.get_buffer_sizes()
        total_buffer_size = input_buf + weight_buf + output_buf
        
        # Energy in pJ (picojoules) - very rough estimate
        energy_per_op = 1.0  # pJ per operation
        buffer_energy = total_buffer_size / 1024 * 10  # 10 pJ per KB access
        energy = total_ops * energy_per_op + buffer_energy * total_ops / 1000
        
        # Utilization: how well PEs are used (simplified)
        # Better dataflows and tiling should have higher utilization
        utilization = 0.5 + 0.3 * (gene.buffer_budget_class / 2.0)
        if gene.dataflow == "RS":
            utilization += 0.1  # RS typically has good utilization
        utilization = min(1.0, utilization)
        
        # Area: rough estimate in mm^2
        area = pe_count * 0.01 + total_buffer_size / (1024 * 1024) * 0.1
        
        return {
            "latency": latency,
            "energy": energy,
            "utilization": utilization,
            "area": area,
        }
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

