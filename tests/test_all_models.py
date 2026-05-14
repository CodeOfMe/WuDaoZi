#!/usr/bin/env python3
"""WuDaoZi Model Benchmark Report Generator.

Tests all downloaded models on ROCm 7900 XTX and generates comprehensive report.
"""

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wudaozi.model_manager import (
    AVAILABLE_MODELS,
    MultiModelEngine,
    _is_rocm,
    _detect_device,
    _get_compute_dtype,
    _get_gpu_vram_gb,
    get_model_dir,
    get_model_size,
    list_downloaded_models,
)

TEST_PROMPT = "A beautiful traditional Chinese landscape painting with mountains, rivers, and a small boat"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "model_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_system_info() -> dict:
    """Collect system and environment information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "is_rocm": _is_rocm(),
        "hip_version": getattr(torch.version, "hip", None),
        "device": _detect_device(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_vram_gb": round(_get_gpu_vram_gb(), 2),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "compute_dtype": str(_get_compute_dtype(_detect_device())),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_properties"] = {
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "multi_processor_count": props.multi_processor_count,
        }
    return info


def get_model_spec(model_key: str) -> dict:
    """Collect detailed model specifications."""
    info = AVAILABLE_MODELS.get(model_key)
    if not info:
        return {"error": f"Model {model_key} not found"}

    model_dir = get_model_dir(model_key)
    size_info = get_model_size(model_key)

    spec = {
        "model_key": model_key,
        "name": info.name,
        "model_type": info.model_type,
        "description": info.description,
        "default_steps": info.default_steps,
        "supports_guidance": info.supports_guidance,
        "quantization": info.quantization,
        "estimated_size_gb": info.size_gb,
        "actual_size_gb": round(size_info.get("size_gb", 0), 2),
        "model_path": str(model_dir),
        "modelscope_id": info.modelscope_id,
        "hf_id": info.hf_id,
    }

    # Read model_index.json if exists
    index_path = model_dir / "model_index.json"
    if index_path.exists():
        try:
            with open(index_path) as f:
                model_index = json.load(f)
            spec["pipeline_class"] = model_index.get("_class_name")
            spec["diffusers_version"] = model_index.get("_diffusers_version")
            spec["components"] = {k: v for k, v in model_index.items() if k not in ["_class_name", "_diffusers_version"]}
        except Exception as e:
            spec["model_index_error"] = str(e)

    # Count files
    file_count = 0
    total_size = 0
    if model_dir.exists():
        for p in model_dir.rglob("*"):
            if p.is_file():
                file_count += 1
                total_size += p.stat().st_size
    spec["file_count"] = file_count
    spec["total_size_bytes"] = total_size
    spec["total_size_gb"] = round(total_size / (1024**3), 2)

    return spec


def test_model_generation(model_key: str, test_prompt: str = TEST_PROMPT) -> dict:
    """Test model generation and record performance metrics."""
    result = {
        "model_key": model_key,
        "success": False,
        "error": None,
        "load_time_s": None,
        "generate_time_s": None,
        "peak_vram_gb": None,
        "output_image": None,
        "image_size": None,
    }

    try:
        print(f"\n{'='*60}")
        print(f"Testing: {model_key}")
        print(f"{'='*60}")

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Load model
        print(f"Loading {model_key}...")
        start_load = time.time()
        engine = MultiModelEngine(model_key=model_key)
        engine.load()
        load_time = time.time() - start_load
        result["load_time_s"] = round(load_time, 2)
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Device: {engine.device}")
        print(f"  dtype: {engine.dtype}")
        print(f"  is_rocm: {engine.is_rocm}")

        # Get peak VRAM after load
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            result["peak_vram_gb"] = round(peak_mem / (1024**3), 2)
            print(f"  Peak VRAM after load: {result['peak_vram_gb']:.2f} GB")

        # Generate image
        print(f"Generating image...")
        start_gen = time.time()
        img = engine.generate(
            test_prompt,
            height=1024,
            width=1024,
            num_inference_steps=8,
            seed=42,
        )
        gen_time = time.time() - start_gen
        result["generate_time_s"] = round(gen_time, 2)
        print(f"  Generate time: {gen_time:.2f}s")

        # Get peak VRAM after generation
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            result["peak_vram_gb"] = round(peak_mem / (1024**3), 2)
            print(f"  Peak VRAM after gen: {result['peak_vram_gb']:.2f} GB")

        # Save image
        output_path = OUTPUT_DIR / f"{model_key}_test.png"
        img.save(output_path)
        result["output_image"] = str(output_path)
        result["image_size"] = list(img.size)
        result["success"] = True
        print(f"  Image saved: {output_path}")
        print(f"  Image size: {img.size}")

    except Exception as e:
        result["error"] = str(e)
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def main():
    print("WuDaoZi Model Benchmark Suite")
    print("="*60)

    # System info
    sys_info = get_system_info()
    print(f"\nSystem Information:")
    print(f"  GPU: {sys_info['gpu_name']}")
    print(f"  VRAM: {sys_info['gpu_vram_gb']} GB")
    print(f"  ROCm: {sys_info['is_rocm']} (HIP {sys_info['hip_version']})")
    print(f"  PyTorch: {sys_info['torch_version']}")
    print(f"  Dtype: {sys_info['compute_dtype']}")

    # Downloaded models
    downloaded = list_downloaded_models()
    print(f"\nDownloaded models: {downloaded}")

    # Test each model
    results = []
    specs = []
    for model_key in downloaded:
        spec = get_model_spec(model_key)
        specs.append(spec)

        result = test_model_generation(model_key)
        results.append(result)

    # Compile report
    report = {
        "system": sys_info,
        "models_tested": len(results),
        "specs": specs,
        "results": results,
    }

    # Save report
    report_path = OUTPUT_DIR / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*60}")
    print(f"Benchmark report saved to: {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        status = "SUCCESS" if r["success"] else f"FAILED: {r['error'][:50]}"
        load_t = f"{r['load_time_s']:.1f}s" if r['load_time_s'] else "N/A"
        gen_t = f"{r['generate_time_s']:.1f}s" if r['generate_time_s'] else "N/A"
        vram = f"{r['peak_vram_gb']:.1f}GB" if r['peak_vram_gb'] else "N/A"
        img = r['output_image'] if r['output_image'] else "N/A"
        print(f"  {r['model_key']}: {status}")
        print(f"    Load: {load_t}, Generate: {gen_t}, Peak VRAM: {vram}")
        print(f"    Image: {img}")

    return report


if __name__ == "__main__":
    main()
