"""Model comparison test - generates same image with all working models."""

import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PROMPT = "A beautiful traditional Chinese landscape painting with mountains, rivers, and a small boat"
SEED = 42
HEIGHT = 1024
WIDTH = 1024


def get_gpu_info():
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "torch": torch.__version__,
        "rocm": hasattr(torch.version, "hip") and torch.version.hip is not None,
        "hip": getattr(torch.version, "hip", None),
    }


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_sdxl_turbo():
    from diffusers import AutoPipelineForText2Image
    clear_memory()
    model_path = PROJECT_ROOT / "models" / "sdxl-turbo"
    print("\n" + "="*60)
    print("Testing SDXL-Turbo")
    print("="*60)
    start = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained(
        str(model_path), torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=True,
    )
    pipe.to("cuda")
    load_time = time.time() - start
    vram_after_load = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  Load time: {load_time:.2f}s, VRAM: {vram_after_load:.2f} GB")
    start = time.time()
    img = pipe(
        prompt=TEST_PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=4,
        guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(SEED),
    ).images[0]
    gen_time = time.time() - start
    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  Generate: {gen_time:.2f}s, Peak VRAM: {vram_peak:.2f} GB")
    output_path = OUTPUT_DIR / "sdxl-turbo.png"
    img.save(output_path)
    return {
        "model": "SDXL-Turbo", "model_key": "sdxl-turbo", "pipeline": "StableDiffusionXLPipeline",
        "load_time": round(load_time, 2), "generate_time": round(gen_time, 2),
        "steps": 4, "guidance_scale": 0.0, "peak_vram_gb": round(vram_peak, 2),
        "output": str(output_path), "image_size": list(img.size), "success": True,
    }


def test_kolors():
    from diffusers import KolorsPipeline
    clear_memory()
    model_path = PROJECT_ROOT / "models" / "kolors"
    print("\n" + "="*60)
    print("Testing Kolors")
    print("="*60)
    start = time.time()
    pipe = KolorsPipeline.from_pretrained(
        str(model_path), torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    pipe.to("cuda")
    load_time = time.time() - start
    vram_after_load = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  Load time: {load_time:.2f}s, VRAM: {vram_after_load:.2f} GB")
    start = time.time()
    img = pipe(
        prompt=TEST_PROMPT, negative_prompt="blurry, low quality, distorted",
        height=HEIGHT, width=WIDTH, num_inference_steps=20, guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(SEED),
    ).images[0]
    gen_time = time.time() - start
    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  Generate: {gen_time:.2f}s, Peak VRAM: {vram_peak:.2f} GB")
    output_path = OUTPUT_DIR / "kolors.png"
    img.save(output_path)
    return {
        "model": "Kolors", "model_key": "kolors", "pipeline": "KolorsPipeline",
        "load_time": round(load_time, 2), "generate_time": round(gen_time, 2),
        "steps": 20, "guidance_scale": 7.5, "peak_vram_gb": round(vram_peak, 2),
        "output": str(output_path), "image_size": list(img.size), "success": True,
    }


def create_comparison_grid(results):
    images, labels = [], []
    for r in results:
        if r["success"]:
            img = Image.open(r["output"])
            images.append(img.resize((512, 512)))
            labels.append(f"{r['model']}\n{r['generate_time']:.1f}s / {r['peak_vram_gb']:.1f}GB")
    n = len(images)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 512, 572
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h + 30), (240, 240, 240))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_title = font
    title = f"Model Comparison | Seed: {SEED} | {HEIGHT}x{WIDTH} | {datetime.now().strftime('%Y-%m-%d')}"
    draw.text((10, 5), title, fill=(0, 0, 0), font=font_title)
    for idx, (img, label) in enumerate(zip(images, labels)):
        col, row = idx % cols, idx // cols
        x, y = col * cell_w, row * cell_h + 30
        grid.paste(img, (x, y))
        draw.rectangle([x, y + 512, x + cell_w, y + cell_h], fill=(50, 50, 50))
        draw.text((x + 10, y + 520), label, fill=(255, 255, 255), font=font)
    output_path = OUTPUT_DIR / "comparison_grid.png"
    grid.save(output_path)
    print(f"\nComparison grid saved: {output_path}")
    return str(output_path)


def main():
    print("WuDaoZi Model Comparison Test")
    print("="*60)
    print(f"Prompt: {TEST_PROMPT}")
    print(f"Seed: {SEED}, Size: {HEIGHT}x{WIDTH}")
    print(f"GPU: {get_gpu_info()['gpu']} (ROCm: {get_gpu_info()['rocm']})")
    results = []
    try:
        results.append(test_sdxl_turbo())
    except Exception as e:
        print(f"SDXL-Turbo FAILED: {e}")
        results.append({"model": "SDXL-Turbo", "success": False, "error": str(e)})
    try:
        results.append(test_kolors())
    except Exception as e:
        print(f"Kolors FAILED: {e}")
        results.append({"model": "Kolors", "success": False, "error": str(e)})
    grid_path = create_comparison_grid(results)
    report = {
        "timestamp": datetime.now().isoformat(), "prompt": TEST_PROMPT,
        "seed": SEED, "size": f"{HEIGHT}x{WIDTH}", "gpu_info": get_gpu_info(),
        "results": results, "comparison_grid": grid_path,
    }
    report_path = OUTPUT_DIR / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {report_path}")
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        if r["success"]:
            print(f"  {r['model']}: {r['generate_time']:.1f}s, {r['peak_vram_gb']:.1f}GB VRAM")
        else:
            print(f"  {r['model']}: FAILED - {r['error']}")


if __name__ == "__main__":
    main()
