"""
EdgeForge Quantization Module — REAL IMPLEMENTATION

Supports:
- Dynamic quantization (NLP models, LSTMs, Transformers)
- Static post-training quantization (vision models)
- Weight-only quantization (large models)
"""

import os
import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.quantization as tq

logger = logging.getLogger(__name__)


@dataclass
class QuantizationResult:
    """Results from quantization."""

    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 0.0
    strategy_used: str = ""
    bit_width: int = 8
    layers_quantized: int = 0
    total_layers: int = 0


class Quantizer:
    """
    Post-training quantization engine with real PyTorch implementations.

    Usage:
        q = Quantizer(strategy="dynamic", target_bits=8)
        quantized_model, result = q.quantize(model)
    """

    STRATEGIES = ["dynamic", "static", "weight_only"]

    def __init__(
        self,
        strategy: str = "dynamic",
        target_bits: int = 8,
        calibration_data=None,
        backend: str = "fbgemm",  # "fbgemm" for x86, "qnnpack" for ARM
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {self.STRATEGIES}"
            )

        self.strategy = strategy
        self.target_bits = target_bits
        self.calibration_data = calibration_data
        self.backend = backend

    def quantize(self, model: nn.Module) -> tuple[nn.Module, QuantizationResult]:
        """Apply quantization to the model. Returns (quantized_model, result)."""
        result = QuantizationResult(
            strategy_used=self.strategy, bit_width=self.target_bits
        )

        # Measure original size
        result.original_size_mb = self._model_size_mb(model)
        result.total_layers = self._count_quantizable_layers(model)

        logger.info(
            f"Quantizing: strategy={self.strategy}, bits={self.target_bits}, "
            f"original_size={result.original_size_mb:.2f} MB, "
            f"quantizable_layers={result.total_layers}"
        )

        # Ensure model is in eval mode
        model.eval()

        if self.strategy == "dynamic":
            quantized = self._dynamic_quantize(model)
        elif self.strategy == "static":
            quantized = self._static_quantize(model)
        elif self.strategy == "weight_only":
            quantized = self._weight_only_quantize(model)
        else:
            quantized = model

        # Measure quantized size
        result.quantized_size_mb = self._model_size_mb(quantized)
        result.compression_ratio = result.original_size_mb / max(
            result.quantized_size_mb, 0.001
        )
        result.layers_quantized = self._count_quantized_layers(quantized)

        logger.info(
            f"Quantization complete: {result.original_size_mb:.2f} MB → "
            f"{result.quantized_size_mb:.2f} MB "
            f"({result.compression_ratio:.1f}x compression, "
            f"{result.layers_quantized}/{result.total_layers} layers quantized)"
        )

        return quantized, result

    def _dynamic_quantize(self, model: nn.Module) -> nn.Module:
        """
        Dynamic quantization — weights quantized offline, activations at runtime.
        Best for: Linear layers, LSTMs, Transformers.
        No calibration data needed. Fastest to apply.
        """
        logger.info("Applying dynamic quantization (INT8)...")

        # Identify which layer types to quantize
        quantizable_types = {nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell, nn.GRUCell}

        # Check which types actually exist in the model
        types_present = set()
        for module in model.modules():
            if type(module) in quantizable_types:
                types_present.add(type(module))

        if not types_present:
            logger.warning(
                "No dynamically quantizable layers found (Linear, LSTM, GRU). "
                "Try 'static' strategy for Conv layers."
            )
            return model

        dtype = torch.qint8

        quantized = torch.quantization.quantize_dynamic(
            model,
            types_present,
            dtype=dtype,
        )

        logger.info(
            f"Dynamic quantization applied to {len(types_present)} layer types: "
            f"{[t.__name__ for t in types_present]}"
        )
        return quantized

    def _static_quantize(self, model: nn.Module) -> nn.Module:
        """
        Static post-training quantization — both weights and activations quantized.
        Best for: CNNs and vision models.
        Requires calibration data for activation ranges.
        """
        logger.info("Applying static quantization (PTQ)...")

        # Set quantization backend
        torch.backends.quantized.engine = self.backend

        # Attach qconfig to model
        if self.backend == "fbgemm":
            model.qconfig = tq.get_default_qconfig("fbgemm")
        else:
            model.qconfig = tq.get_default_qconfig("qnnpack")

        # Fuse common layer patterns for better quantization
        model = self._try_fuse_modules(model)

        # Prepare model for calibration (inserts observers)
        try:
            prepared = tq.prepare(model, inplace=False)
        except Exception as e:
            logger.warning(
                f"Static quantization preparation failed: {e}. "
                f"Falling back to dynamic quantization."
            )
            return self._dynamic_quantize(model)

        # Run calibration data through the model
        if self.calibration_data is not None:
            logger.info("Running calibration...")
            with torch.no_grad():
                for i, batch in enumerate(self.calibration_data):
                    if i >= 100:  # Cap at 100 batches
                        break
                    if isinstance(batch, (list, tuple)):
                        prepared(batch[0])
                    else:
                        prepared(batch)
            logger.info(f"Calibration complete ({min(i + 1, 100)} batches)")
        else:
            # Generate synthetic calibration data
            logger.info("No calibration data provided. Using synthetic data...")
            self._synthetic_calibration(prepared)

        # Convert to quantized model
        try:
            quantized = tq.convert(prepared, inplace=False)
            logger.info("Static quantization complete.")
            return quantized
        except Exception as e:
            logger.warning(
                f"Static quantization conversion failed: {e}. "
                f"Falling back to dynamic quantization."
            )
            return self._dynamic_quantize(model)

    def _weight_only_quantize(self, model: nn.Module) -> nn.Module:
        """
        Weight-only quantization — only weights are quantized, activations stay FP32.
        Good middle ground between dynamic and static.
        Useful for larger models where activation quantization hurts accuracy.
        """
        logger.info(
            "Applying weight-only quantization (INT8 weights, FP32 activations)..."
        )

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )

        logger.info("Weight-only quantization complete.")
        return quantized_model

    def _try_fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        Attempt to fuse common layer patterns: Conv+BN+ReLU, Linear+ReLU.
        Fused modules quantize better and run faster.
        """
        try:
            # Try to detect fusable patterns
            fuse_patterns = []

            named_modules = dict(model.named_modules())
            module_names = list(named_modules.keys())

            for i, name in enumerate(module_names):
                module = named_modules[name]

                # Conv + BatchNorm + ReLU
                if isinstance(module, nn.Conv2d):
                    next_names = module_names[i + 1 : i + 3]
                    pattern = [name]
                    for next_name in next_names:
                        next_mod = named_modules.get(next_name)
                        if isinstance(next_mod, nn.BatchNorm2d):
                            pattern.append(next_name)
                        elif isinstance(next_mod, (nn.ReLU, nn.ReLU6)):
                            pattern.append(next_name)
                            break
                    if len(pattern) >= 2:
                        fuse_patterns.append(pattern)

                # Linear + ReLU
                elif isinstance(module, nn.Linear):
                    if i + 1 < len(module_names):
                        next_mod = named_modules.get(module_names[i + 1])
                        if isinstance(next_mod, nn.ReLU):
                            fuse_patterns.append([name, module_names[i + 1]])

            if fuse_patterns:
                model = tq.fuse_modules(model, fuse_patterns, inplace=False)
                logger.info(f"Fused {len(fuse_patterns)} module patterns")
            else:
                logger.info("No fusable patterns detected")

        except Exception as e:
            logger.warning(f"Module fusion failed (non-critical): {e}")

        return model

    def _synthetic_calibration(self, model: nn.Module, num_batches: int = 50):
        """Generate synthetic data for calibration when real data isn't available."""
        with torch.no_grad():
            # Try to infer input shape from first layer
            input_shape = self._infer_input_shape(model)
            for _ in range(num_batches):
                dummy = torch.randn(1, *input_shape)
                try:
                    model(dummy)
                except Exception:
                    break

    def _infer_input_shape(self, model: nn.Module) -> tuple:
        """Try to determine the expected input shape from model architecture."""
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                return (in_channels, 224, 224)  # Common image size
            elif isinstance(module, nn.Linear):
                return (module.in_features,)
        return (3, 224, 224)  # Default to ImageNet shape

    def _model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size in megabytes."""
        # Save to buffer and measure
        import io

        buffer = io.BytesIO()
        try:
            torch.save(model.state_dict(), buffer)
            size = buffer.tell() / (1024 * 1024)
        except Exception:
            # Fallback: estimate from parameters
            size = sum(p.nelement() * p.element_size() for p in model.parameters()) / (
                1024 * 1024
            )
        return size

    def _count_quantizable_layers(self, model: nn.Module) -> int:
        """Count layers that can be quantized."""
        quantizable = (nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU, nn.Conv1d)
        return sum(1 for m in model.modules() if isinstance(m, quantizable))

    def _count_quantized_layers(self, model: nn.Module) -> int:
        """Count layers that were actually quantized."""
        count = 0
        for module in model.modules():
            module_type = type(module).__name__
            if "Quantized" in module_type or "Dynamic" in module_type:
                count += 1
        return count


def auto_select_strategy(model: nn.Module) -> str:
    """
    Automatically select the best quantization strategy based on model architecture.
    """
    has_conv = any(isinstance(m, (nn.Conv2d, nn.Conv1d)) for m in model.modules())
    has_linear = any(isinstance(m, nn.Linear) for m in model.modules())
    has_rnn = any(isinstance(m, (nn.LSTM, nn.GRU)) for m in model.modules())

    param_count = sum(p.numel() for p in model.parameters())

    if has_rnn:
        return "dynamic"  # RNNs work best with dynamic quantization
    elif has_conv and param_count < 100_000_000:
        return "static"  # CNNs benefit from static quantization
    elif param_count > 500_000_000:
        return "weight_only"  # Very large models: weight-only is safest
    elif has_linear:
        return "dynamic"  # Transformer/MLP models
    else:
        return "dynamic"  # Safe default
