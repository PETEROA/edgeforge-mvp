"""
EdgeForge Optimization Pipeline — REAL IMPLEMENTATION

End-to-end pipeline: Load → Analyze → Prune → Quantize → Export → Benchmark
"""

import os
import io
import time
import json
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .quantizer import Quantizer, auto_select_strategy
from .pruner import Pruner

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Complete results from an optimization run."""

    # Size
    original_size_mb: float = 0.0
    optimized_size_mb: float = 0.0
    compression_ratio: float = 0.0

    # Speed (estimated)
    original_latency_ms: float = 0.0
    optimized_latency_ms: float = 0.0
    speedup_factor: float = 0.0

    # Accuracy (if evaluation data provided)
    accuracy_original: float = 0.0
    accuracy_optimized: float = 0.0
    accuracy_drop: float = 0.0

    # Model info
    original_params: int = 0
    optimized_params: int = 0
    sparsity: float = 0.0

    # Paths
    optimized_model_path: str = ""
    benchmark_report_path: str = ""

    # Pipeline info
    stages_completed: list[str] = field(default_factory=list)
    stage_details: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    total_time_seconds: float = 0.0


# Device profile latency multipliers (ms per million parameters)
DEVICE_LATENCY_FACTOR = {
    "android-low": 0.8,
    "android-mid": 0.3,
    "rpi4": 0.5,
    "rpi5": 0.25,
    "jetson-nano": 0.12,
    "jetson-orin": 0.05,
    "edge-server": 0.03,
    "browser-wasm": 1.2,
    "ios-coreml": 0.15,
    "tinyml": 3.0,
}


class OptimizationPipeline:
    """
    Main optimization pipeline that orchestrates compression stages.

    Usage:
        pipeline = OptimizationPipeline(
            model_path="/path/to/model.pt",
            device_profile="android-mid",
        )
        result = pipeline.run()
    """

    def __init__(
        self,
        model_path: str | None = None,
        hf_model_id: str | None = None,
        device_profile: str = "android-mid",
        output_dir: str = "./storage/optimized",
        enable_quantization: bool = True,
        enable_pruning: bool = True,
        enable_distillation: bool = False,
        target_size_mb: float | None = None,
        max_accuracy_drop: float = 0.02,
        # Advanced options
        quantization_strategy: str = "auto",
        pruning_method: str = "structured",
        pruning_sparsity: float = 0.5,
        pruning_criterion: str = "l1_norm",
        calibration_data=None,
    ):
        self.model_path = model_path
        self.hf_model_id = hf_model_id
        self.device_profile = device_profile
        self.output_dir = output_dir
        self.enable_quantization = enable_quantization
        self.enable_pruning = enable_pruning
        self.enable_distillation = enable_distillation
        self.target_size_mb = target_size_mb
        self.max_accuracy_drop = max_accuracy_drop

        self.quantization_strategy = quantization_strategy
        self.pruning_method = pruning_method
        self.pruning_sparsity = pruning_sparsity
        self.pruning_criterion = pruning_criterion
        self.calibration_data = calibration_data

        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> OptimizationResult:
        """Execute the full optimization pipeline."""
        result = OptimizationResult()
        start_time = time.time()

        logger.info(f"{'=' * 60}")
        logger.info(f"EdgeForge Optimization Pipeline")
        logger.info(f"Device target: {self.device_profile}")
        logger.info(f"Quantization: {self.enable_quantization}")
        logger.info(
            f"Pruning: {self.enable_pruning} (method={self.pruning_method}, sparsity={self.pruning_sparsity})"
        )
        logger.info(f"Distillation: {self.enable_distillation}")
        logger.info(f"{'=' * 60}")

        try:
            # ---- Stage 1: Load & Analyze ----
            logger.info("\n[Stage 1/5] Loading and analyzing model...")
            model = self._load_model()
            result.original_size_mb = self._model_size_mb(model)
            result.original_params = sum(p.numel() for p in model.parameters())
            result.stages_completed.append("analyze")

            result.stage_details["analyze"] = {
                "size_mb": round(result.original_size_mb, 2),
                "params": result.original_params,
                "architecture": type(model).__name__,
            }

            logger.info(
                f"  Model loaded: {type(model).__name__}, "
                f"{result.original_params:,} params, "
                f"{result.original_size_mb:.2f} MB"
            )

            # ---- Stage 2: Pruning ----
            if self.enable_pruning:
                logger.info(
                    f"\n[Stage 2/5] Pruning ({self.pruning_method}, {self.pruning_sparsity:.0%} sparsity)..."
                )

                pruner = Pruner(
                    method=self.pruning_method,
                    sparsity=self.pruning_sparsity,
                    criterion=self.pruning_criterion,
                )
                model, prune_result = pruner.prune(model)
                result.stages_completed.append("prune")

                result.stage_details["prune"] = {
                    "method": prune_result.method_used,
                    "target_sparsity": prune_result.target_sparsity,
                    "actual_sparsity": round(prune_result.actual_sparsity, 4),
                    "layers_pruned": prune_result.layers_pruned,
                    "size_after_mb": round(prune_result.pruned_size_mb, 2),
                }

                logger.info(
                    f"  Pruning done: {prune_result.layers_pruned} layers, "
                    f"sparsity={prune_result.actual_sparsity:.1%}"
                )
            else:
                logger.info("\n[Stage 2/5] Pruning: SKIPPED")

            # ---- Stage 3: Quantization ----
            if self.enable_quantization:
                # Auto-select strategy if needed
                strategy = self.quantization_strategy
                if strategy == "auto":
                    strategy = auto_select_strategy(model)
                    logger.info(
                        f"\n[Stage 3/5] Quantization (auto-selected: {strategy})..."
                    )
                else:
                    logger.info(f"\n[Stage 3/5] Quantization ({strategy})...")

                # Select backend based on device profile
                backend = self._select_quant_backend()

                quantizer = Quantizer(
                    strategy=strategy,
                    target_bits=8,
                    calibration_data=self.calibration_data,
                    backend=backend,
                )
                model, quant_result = quantizer.quantize(model)
                result.stages_completed.append("quantize")

                result.stage_details["quantize"] = {
                    "strategy": quant_result.strategy_used,
                    "bit_width": quant_result.bit_width,
                    "layers_quantized": quant_result.layers_quantized,
                    "compression_ratio": round(quant_result.compression_ratio, 2),
                    "size_after_mb": round(quant_result.quantized_size_mb, 2),
                }

                logger.info(
                    f"  Quantization done: {quant_result.layers_quantized} layers, "
                    f"{quant_result.compression_ratio:.1f}x compression"
                )
            else:
                logger.info("\n[Stage 3/5] Quantization: SKIPPED")

            # ---- Stage 4: Knowledge Distillation ----
            if self.enable_distillation:
                logger.info("\n[Stage 4/5] Knowledge Distillation...")
                # TODO: Implement in Week 5-6
                # Will use the original (unpruned/unquantized) model as teacher
                logger.warning("  Distillation not yet implemented — skipping")
            else:
                logger.info("\n[Stage 4/5] Distillation: SKIPPED")

            # ---- Stage 5: Export ----
            logger.info("\n[Stage 5/5] Exporting optimized model...")
            output_path = self._export(model)
            result.optimized_model_path = output_path
            result.optimized_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            result.stages_completed.append("export")

            # ---- Calculate Final Metrics ----
            result.compression_ratio = result.original_size_mb / max(
                result.optimized_size_mb, 0.001
            )
            result.optimized_params = sum(p.numel() for p in model.parameters())
            result.sparsity = 1.0 - (
                result.optimized_params / max(result.original_params, 1)
            )

            # Estimate latency
            latency_factor = DEVICE_LATENCY_FACTOR.get(self.device_profile, 0.3)
            result.original_latency_ms = (
                (result.original_params / 1_000_000) * latency_factor * 1000
            )
            result.optimized_latency_ms = result.original_latency_ms / max(
                result.compression_ratio, 1.0
            )
            result.speedup_factor = result.original_latency_ms / max(
                result.optimized_latency_ms, 0.001
            )

            # Generate benchmark report
            result.benchmark_report_path = self._generate_report(result)

            result.total_time_seconds = time.time() - start_time

            # ---- Summary ----
            logger.info(f"\n{'=' * 60}")
            logger.info(f"OPTIMIZATION COMPLETE")
            logger.info(f"{'=' * 60}")
            logger.info(
                f"  Size:        {result.original_size_mb:.2f} MB → {result.optimized_size_mb:.2f} MB ({result.compression_ratio:.1f}x)"
            )
            logger.info(
                f"  Params:      {result.original_params:,} → {result.optimized_params:,}"
            )
            logger.info(
                f"  Est. Latency: {result.original_latency_ms:.1f} ms → {result.optimized_latency_ms:.1f} ms ({result.speedup_factor:.1f}x)"
            )
            logger.info(f"  Stages:      {' → '.join(result.stages_completed)}")
            logger.info(f"  Time:        {result.total_time_seconds:.1f}s")
            logger.info(f"  Output:      {result.optimized_model_path}")
            logger.info(f"  Report:      {result.benchmark_report_path}")
            logger.info(f"{'=' * 60}")

        except Exception as e:
            result.errors.append(str(e))
            result.total_time_seconds = time.time() - start_time
            logger.error(
                f"Optimization failed after {result.total_time_seconds:.1f}s: {e}"
            )
            raise

        return result

    def _load_model(self) -> nn.Module:
        """Load model from file path or HuggingFace."""
        if self.hf_model_id:
            return self._load_from_huggingface(self.hf_model_id)

        if self.model_path:
            logger.info(f"  Loading from: {self.model_path}")

            if self.model_path.endswith(".onnx"):
                raise NotImplementedError(
                    "ONNX optimization pipeline in development. "
                    "For now, please provide PyTorch (.pt/.pth) models."
                )

            model = torch.load(self.model_path, map_location="cpu", weights_only=False)

            # Handle case where saved file is a state_dict, not a full model
            if isinstance(model, dict) and not isinstance(model, nn.Module):
                raise ValueError(
                    "File contains a state_dict, not a full model. "
                    "Please save with torch.save(model, path) not torch.save(model.state_dict(), path)"
                )

            if isinstance(model, nn.Module):
                model.eval()
                return model

            raise ValueError(f"Unexpected model type: {type(model)}")

        raise ValueError("Either model_path or hf_model_id must be provided")

    def _load_from_huggingface(self, model_id: str) -> nn.Module:
        """Load a model from HuggingFace Hub."""
        logger.info(f"  Loading from HuggingFace: {model_id}")

        try:
            from transformers import AutoModel, AutoModelForImageClassification

            # Try vision model first, fall back to generic
            try:
                model = AutoModelForImageClassification.from_pretrained(model_id)
            except Exception:
                model = AutoModel.from_pretrained(model_id)

            model.eval()
            return model

        except ImportError:
            raise ImportError(
                "transformers package required for HuggingFace loading. "
                "Install with: pip install transformers"
            )

    def _select_quant_backend(self) -> str:
        """Select quantization backend based on target device."""
        arm_devices = {
            "android-low",
            "android-mid",
            "rpi4",
            "rpi5",
            "jetson-nano",
            "jetson-orin",
            "ios-coreml",
            "tinyml",
        }
        if self.device_profile in arm_devices:
            return "qnnpack"
        return "fbgemm"

    def _model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size by serializing to buffer."""
        buffer = io.BytesIO()
        try:
            torch.save(model.state_dict(), buffer)
            return buffer.tell() / (1024 * 1024)
        except Exception:
            return sum(p.nelement() * p.element_size() for p in model.parameters()) / (
                1024 * 1024
            )

    def _export(self, model: nn.Module) -> str:
        """Export the optimized model."""
        timestamp = int(time.time())
        filename = f"optimized_{timestamp}.pt"
        output_path = os.path.join(self.output_dir, filename)

        torch.save(model, output_path)
        logger.info(f"  Model saved: {output_path}")

        return output_path

    def _generate_report(self, result: OptimizationResult) -> str:
        """Generate a JSON benchmark report."""
        report_dir = os.path.join(os.path.dirname(self.output_dir), "reports")
        os.makedirs(report_dir, exist_ok=True)

        timestamp = int(time.time())
        report_path = os.path.join(report_dir, f"benchmark_{timestamp}.json")

        report = {
            "edgeforge_benchmark": {
                "version": "1.0",
                "device_target": self.device_profile,
                "pipeline": {
                    "quantization": self.enable_quantization,
                    "pruning": self.enable_pruning,
                    "distillation": self.enable_distillation,
                },
                "results": {
                    "size": {
                        "original_mb": round(result.original_size_mb, 2),
                        "optimized_mb": round(result.optimized_size_mb, 2),
                        "compression_ratio": round(result.compression_ratio, 2),
                        "reduction_pct": round(
                            (
                                1
                                - result.optimized_size_mb
                                / max(result.original_size_mb, 0.001)
                            )
                            * 100,
                            1,
                        ),
                    },
                    "parameters": {
                        "original": result.original_params,
                        "optimized": result.optimized_params,
                        "sparsity": round(result.sparsity, 4),
                    },
                    "latency_estimate": {
                        "original_ms": round(result.original_latency_ms, 1),
                        "optimized_ms": round(result.optimized_latency_ms, 1),
                        "speedup_factor": round(result.speedup_factor, 1),
                        "note": "Estimated based on device profile. Run on target hardware for actual measurements.",
                    },
                },
                "stages": result.stage_details,
                "total_time_seconds": round(result.total_time_seconds, 2),
            }
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report_path
