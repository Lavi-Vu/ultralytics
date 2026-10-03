# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Benchmark LightEdgeDet ONNX, NCNN, and Hailo models on Raspberry Pi 5.

Each configuration runs in a fresh process so model allocations and Hailo device handles do not leak into the next
measurement. Latency is Ultralytics' model-inference time and excludes preprocessing/postprocessing. Power is average
Pi input power sampled from ``vcgencmd pmic_read_adc``; use ``--power-file`` for an external sensor when unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import resource
import shutil
import statistics
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkSpec:
    """One model configuration in the requested report."""

    platform: str
    runtime: str
    image_size: int
    model: str


@dataclass
class BenchmarkResult:
    """Measured values for one benchmark configuration."""

    platform: str
    runtime: str
    image_size: int
    latency_ms: float
    fps: float
    peak_memory_mb: float
    power_w: float | None
    p50_ms: float
    p95_ms: float
    samples: int


def parse_args() -> argparse.Namespace:
    """Parse benchmark and internal worker arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Image used for every timed inference")
    parser.add_argument("--onnx-640", type=Path, help="640x640 ONNX model")
    parser.add_argument("--onnx-512", type=Path, help="512x512 ONNX model")
    parser.add_argument("--ncnn-640", type=Path, help="640x640 NCNN model directory")
    parser.add_argument("--ncnn-512", type=Path, help="512x512 NCNN model directory")
    parser.add_argument("--hailo-640", type=Path, help="640x640 Hailo model directory")
    parser.add_argument("--hailo-512", type=Path, help="512x512 Hailo model directory")
    parser.add_argument("--skip-hailo", action="store_true", help="Skip both HailoRT benchmark configurations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per model")
    parser.add_argument("--iterations", type=int, default=50, help="Measured iterations per model")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold")
    parser.add_argument("--output", type=Path, default=Path("benchmark_pi"), help="Output path stem")
    parser.add_argument("--power-file", type=Path, help="Optional sensor file containing power in watts")
    parser.add_argument("--power-scale", type=float, default=1.0, help="Multiplier applied to --power-file values")
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    return parser.parse_args()


def read_power_w(power_file: Path | None, power_scale: float) -> float | None:
    """Read total system power from an external sensor or the Raspberry Pi 5 PMIC."""
    if power_file:
        try:
            return float(power_file.read_text().strip()) * power_scale
        except (OSError, ValueError):
            return None
    if not shutil.which("vcgencmd"):
        return None
    reading = subprocess.run(
        ["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2, check=False
    ).stdout
    voltage = re.search(r"EXT5V_V[^\n]*?([0-9]+(?:\.[0-9]+)?)", reading)
    current = re.search(r"EXT5V_A[^\n]*?([0-9]+(?:\.[0-9]+)?)", reading)
    return float(voltage.group(1)) * float(current.group(1)) if voltage and current else None


class PowerSampler:
    """Sample average platform power while measured inference is active."""

    def __init__(self, power_file: Path | None, power_scale: float, interval: float = 0.2):
        self.power_file, self.power_scale, self.interval = power_file, power_scale, interval
        self.samples: list[float] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            value = read_power_w(self.power_file, self.power_scale)
            if value is not None and math.isfinite(value):
                self.samples.append(value)
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=2)

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.samples) if self.samples else None


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile from a non-empty list."""
    ordered = sorted(values)
    return ordered[min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)]


def peak_memory_mb() -> float:
    """Return peak resident memory for this worker process in MiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024  # Linux reports KiB; this script targets Raspberry Pi OS.


def run_worker(args: argparse.Namespace, spec: BenchmarkSpec) -> BenchmarkResult:
    """Load and benchmark one model inside an isolated process."""
    from ultralytics import YOLO

    model = YOLO(spec.model, task="lightedgedet")
    predict_args = {"source": str(args.source), "imgsz": spec.image_size, "conf": args.conf, "verbose": False}
    for _ in range(args.warmup):
        model.predict(**predict_args)

    latencies = []
    with PowerSampler(args.power_file, args.power_scale) as power:
        for _ in range(args.iterations):
            result = model.predict(**predict_args)[0]
            latencies.append(float(result.speed["inference"]))
    mean_latency = statistics.fmean(latencies)
    return BenchmarkResult(
        platform=spec.platform,
        runtime=spec.runtime,
        image_size=spec.image_size,
        latency_ms=mean_latency,
        fps=1000 / mean_latency,
        peak_memory_mb=peak_memory_mb(),
        power_w=power.mean,
        p50_ms=statistics.median(latencies),
        p95_ms=percentile(latencies, 0.95),
        samples=len(latencies),
    )


def benchmark_specs(args: argparse.Namespace) -> list[BenchmarkSpec]:
    """Build the requested benchmark configurations."""
    values = (
        ("Raspberry Pi 5", "ONNX", 640, args.onnx_640),
        ("Raspberry Pi 5", "ONNX", 512, args.onnx_512),
        ("Raspberry Pi 5", "NCNN", 640, args.ncnn_640),
        ("Raspberry Pi 5", "NCNN", 512, args.ncnn_512),
        ("Raspberry Pi 5 + Hailo-8L", "HailoRT INT8", 640, args.hailo_640),
        ("Raspberry Pi 5 + Hailo-8L", "HailoRT INT8", 512, args.hailo_512),
    )
    options = ("--onnx-640", "--onnx-512", "--ncnn-640", "--ncnn-512", "--hailo-640", "--hailo-512")
    if args.skip_hailo:
        values, options = values[:4], options[:4]
    missing = [option for option, value in zip(options, (x[3] for x in values)) if value is None]
    if missing:
        raise SystemExit(f"Missing required model paths: {', '.join(missing)}")
    return [BenchmarkSpec(platform, runtime, size, str(model)) for platform, runtime, size, model in values]


def run_isolated(args: argparse.Namespace, spec: BenchmarkSpec) -> BenchmarkResult:
    """Run a benchmark worker and extract its machine-readable result."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source",
        str(args.source),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--conf",
        str(args.conf),
        "--power-scale",
        str(args.power_scale),
        "--worker-spec",
        json.dumps(asdict(spec)),
    ]
    if args.power_file:
        command.extend(("--power-file", str(args.power_file)))
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    marker = "BENCHMARK_RESULT="
    payload = next((line.removeprefix(marker) for line in process.stdout.splitlines() if line.startswith(marker)), None)
    if process.returncode or payload is None:
        raise RuntimeError(
            f"Benchmark failed for {spec.runtime} {spec.image_size}:\n{process.stdout}\n{process.stderr}"
        )
    return BenchmarkResult(**json.loads(payload))


def format_value(value: float | None, digits: int = 1) -> str:
    """Format a numeric result or an unavailable marker."""
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_table(results: list[BenchmarkResult]) -> str:
    """Render the requested results table in Markdown."""
    lines = [
        "| Platform | Runtime | Image size | Latency (ms) | FPS | Peak memory (MB) | Power (W) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r.platform} | {r.runtime} | {r.image_size} × {r.image_size} | {r.latency_ms:.1f} | "
        f"{r.fps:.1f} | {r.peak_memory_mb:.1f} | {format_value(r.power_w, 2)} |"
        for r in results
    )
    return "\n".join(lines) + "\n"


def save_results(results: list[BenchmarkResult], output: Path) -> None:
    """Write Markdown, CSV, and JSON reports."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".md").write_text(markdown_table(results))
    output.with_suffix(".json").write_text(json.dumps([asdict(x) for x in results], indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=asdict(results[0]))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def main() -> None:
    """Run a worker or orchestrate the full Raspberry Pi benchmark suite."""
    args = parse_args()
    if args.worker_spec:
        result = run_worker(args, BenchmarkSpec(**json.loads(args.worker_spec)))
        print(f"BENCHMARK_RESULT={json.dumps(asdict(result))}")
        return

    results = []
    for spec in benchmark_specs(args):
        print(f"Benchmarking {spec.runtime} at {spec.image_size} × {spec.image_size}...")
        result = run_isolated(args, spec)
        results.append(result)
        print(f"  {result.latency_ms:.1f} ms, {result.fps:.1f} FPS")
    save_results(results, args.output)
    print("\n" + markdown_table(results), end="")
    print(
        f"Saved {args.output.with_suffix('.md')}, {args.output.with_suffix('.csv')}, and {args.output.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
