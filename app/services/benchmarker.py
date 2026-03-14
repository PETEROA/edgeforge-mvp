"""
Benchmark service — profiles original vs. optimized models
and generates comparison reports.
"""

import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    model_name: str = ""
    device_profile: str = ""

    # Size
    original_size_mb: float = 0.0
    optimized_size_mb: float = 0.0
    size_reduction_pct: float = 0.0

    # Speed
    original_latency_ms: float = 0.0
    optimized_latency_ms: float = 0.0
    speedup_factor: float = 0.0

    # Accuracy
    accuracy_original: float = 0.0
    accuracy_optimized: float = 0.0
    accuracy_drop_pct: float = 0.0

    # Resource
    peak_ram_mb: float = 0.0

    # Meta
    stages_applied: list[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if self.stages_applied is None:
            self.stages_applied = []
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class BenchmarkService:
    """
    Profiles models and generates benchmark reports.

    Usage:
        svc = BenchmarkService()
        metrics = svc.profile(
            original_path="model.pt",
            optimized_path="optimized.pt",
            device_profile="android-mid",
        )
        report_path = svc.generate_report(metrics, output_dir="./storage/reports")
    """

    def profile(
        self,
        original_path: str,
        optimized_path: str,
        device_profile: str = "android-mid",
        model_name: str = "model",
        stages_applied: list[str] = None,
    ) -> BenchmarkMetrics:
        """Profile original vs optimized model and return metrics."""
        metrics = BenchmarkMetrics(
            model_name=model_name,
            device_profile=device_profile,
            stages_applied=stages_applied or [],
        )

        # Size comparison
        if os.path.exists(original_path):
            metrics.original_size_mb = os.path.getsize(original_path) / (1024 * 1024)
        if os.path.exists(optimized_path):
            metrics.optimized_size_mb = os.path.getsize(optimized_path) / (1024 * 1024)

        if metrics.original_size_mb > 0:
            metrics.size_reduction_pct = (
                (metrics.original_size_mb - metrics.optimized_size_mb)
                / metrics.original_size_mb * 100
            )

        # Latency profiling
        # TODO (Week 5-6): Implement actual inference timing
        # For now, estimate based on compression ratio
        metrics.original_latency_ms = self._estimate_latency(metrics.original_size_mb, device_profile)
        metrics.optimized_latency_ms = self._estimate_latency(metrics.optimized_size_mb, device_profile)

        if metrics.optimized_latency_ms > 0:
            metrics.speedup_factor = metrics.original_latency_ms / metrics.optimized_latency_ms

        # Accuracy evaluation
        # TODO (Week 5-6): Implement actual accuracy evaluation with test dataset
        metrics.accuracy_original = 0.0
        metrics.accuracy_optimized = 0.0
        metrics.accuracy_drop_pct = 0.0

        return metrics

    def _estimate_latency(self, size_mb: float, device_profile: str) -> float:
        """
        Rough latency estimate based on model size and device.
        Will be replaced with actual profiling in Week 5-6.
        """
        # Base latency per MB for each device class (ms)
        device_speed = {
            "android-low": 15.0,
            "android-mid": 5.0,
            "rpi4": 8.0,
            "rpi5": 4.0,
            "jetson-nano": 2.0,
            "jetson-orin": 0.8,
            "edge-server": 0.5,
            "browser-wasm": 20.0,
            "ios-coreml": 3.0,
            "tinyml": 50.0,
        }
        ms_per_mb = device_speed.get(device_profile, 5.0)
        return size_mb * ms_per_mb

    def generate_report(
        self,
        metrics: BenchmarkMetrics,
        output_dir: str = "./storage/reports",
    ) -> str:
        """Generate a JSON benchmark report. Returns the file path."""
        os.makedirs(output_dir, exist_ok=True)

        report = {
            "edgeforge_benchmark_report": {
                "version": "1.0",
                "generated_at": metrics.timestamp,
                "summary": {
                    "model": metrics.model_name,
                    "target_device": metrics.device_profile,
                    "stages_applied": metrics.stages_applied,
                },
                "size": {
                    "original_mb": round(metrics.original_size_mb, 2),
                    "optimized_mb": round(metrics.optimized_size_mb, 2),
                    "reduction_pct": round(metrics.size_reduction_pct, 1),
                },
                "latency": {
                    "original_ms": round(metrics.original_latency_ms, 1),
                    "optimized_ms": round(metrics.optimized_latency_ms, 1),
                    "speedup": round(metrics.speedup_factor, 1),
                    "note": "Estimated. Actual profiling available in v0.2.",
                },
                "accuracy": {
                    "original": metrics.accuracy_original,
                    "optimized": metrics.accuracy_optimized,
                    "drop_pct": metrics.accuracy_drop_pct,
                    "note": "Requires test dataset. Upload via /v1/benchmarks/evaluate.",
                },
            }
        }

        filename = f"benchmark_{metrics.model_name}_{int(time.time())}.json"
        path = os.path.join(output_dir, filename)

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Benchmark report saved: {path}")
        return path
