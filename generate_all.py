"""WuDaoZi full generation script - generate all images with quantized model."""

import json
import os
import sys
from pathlib import Path

# Correct model path
MODEL_PATH = "/home/fred/Documents/GitHub/Others/Z-Image/Z-Image-Turbo"

# Output directories
OUTPUT_DIR = "/home/fred/Documents/GitHub/Others/Z-Image/WuDaoZi/full_output"
BATCH_OUTPUT = os.path.join(OUTPUT_DIR, "batch")
SERIES_OUTPUT = os.path.join(OUTPUT_DIR, "series")

# Settings optimized for RTX 4060 8GB with quantization
CONFIG = {
    "model_path": MODEL_PATH,
    "low_vram": True,  # Enable 8-bit quantization + sequential CPU offload
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
}


def log(msg: str):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def read_prompts(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def verify_image(path: str) -> bool:
    """Verify that an image file exists and is valid."""
    from PIL import Image
    import numpy as np

    if not Path(path).exists():
        print(f"  ERROR: File not found: {path}")
        return False
    try:
        img = Image.open(path)
        arr = np.array(img)
        mean_val = arr.mean()
        std_val = arr.std()
        print(f"  OK: {path} ({img.size[0]}x{img.size[1]}, mean={mean_val:.1f}, std={std_val:.1f})")
        if mean_val < 10 and std_val < 10:
            print(f"  WARNING: Image may be blank!")
            return False
        return True
    except Exception as e:
        print(f"  ERROR: Invalid image {path}: {e}")
        return False


def generate_single(prompt: str, output_path: str, seed: int) -> bool:
    """Generate a single image."""
    from wudaozi.api import generate_image

    result = generate_image(
        prompt=prompt,
        output=output_path,
        height=CONFIG["height"],
        width=CONFIG["width"],
        num_inference_steps=CONFIG["num_inference_steps"],
        guidance_scale=CONFIG["guidance_scale"],
        seed=seed,
        model_path=CONFIG["model_path"],
        low_vram=CONFIG["low_vram"],
    )

    if result.success:
        print(f"  [OK] {output_path}")
        return verify_image(output_path)
    else:
        print(f"  [FAIL] {result.error}")
        return False


def generate_batch(prompts: list[str], output_dir: str, start_seed: int = 42) -> list[str]:
    """Batch generate images."""
    from wudaozi.api import batch_generate
    import tempfile

    # Create temp prompts file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")
        prompts_file = f.name

    result = batch_generate(
        prompts_file=prompts_file,
        output_dir=output_dir,
        height=CONFIG["height"],
        width=CONFIG["width"],
        num_inference_steps=CONFIG["num_inference_steps"],
        guidance_scale=CONFIG["guidance_scale"],
        start_seed=start_seed,
        model_path=CONFIG["model_path"],
        low_vram=CONFIG["low_vram"],
    )

    os.unlink(prompts_file)

    if result.success:
        print(f"  Generated {result.data['count']} images")
        all_ok = True
        for path in result.data["paths"]:
            if not verify_image(path):
                all_ok = False
        return result.data["paths"] if all_ok else []
    else:
        print(f"  FAILED: {result.error}")
        return []


def generate_series(name: str, theme: str, prompts: list[str], output_dir: str, start_seed: int = 42) -> bool:
    """Generate a series."""
    from wudaozi.api import create_series
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")
        prompts_file = f.name

    result = create_series(
        name=name,
        theme=theme,
        prompts_file=prompts_file,
        output_dir=output_dir,
        height=CONFIG["height"],
        width=CONFIG["width"],
        num_inference_steps=CONFIG["num_inference_steps"],
        guidance_scale=CONFIG["guidance_scale"],
        start_seed=start_seed,
        model_path=CONFIG["model_path"],
        low_vram=CONFIG["low_vram"],
    )

    os.unlink(prompts_file)

    if result.success:
        print(f"  Series '{name}' created: {result.data.get('output_dir')}")
        print(f"  Works: {len(result.data.get('works', []))}")
        all_ok = True
        for work in result.data.get("works", []):
            if not verify_image(work["path"]):
                all_ok = False
        return all_ok
    else:
        print(f"  FAILED: {result.error}")
        return False


def main():
    log("WuDaoZi Full Generation - All Images")
    log(f"Model: {MODEL_PATH}")
    log(f"Settings: {CONFIG['height']}x{CONFIG['width']}, {CONFIG['num_inference_steps']} steps, 8-bit quantization + CPU offload")

    # Verify model
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    # Create output dirs
    os.makedirs(BATCH_OUTPUT, exist_ok=True)
    os.makedirs(SERIES_OUTPUT, exist_ok=True)

    # Read prompts
    test_prompts = read_prompts("/home/fred/Documents/GitHub/Others/Z-Image/WuDaoZi/test_prompts.txt")
    prompts = read_prompts("/home/fred/Documents/GitHub/Others/Z-Image/WuDaoZi/prompts.txt")

    print(f"\nTest prompts: {len(test_prompts)}")
    for i, p in enumerate(test_prompts, 1):
        print(f"  {i}. {p[:50]}...")

    print(f"\nMain prompts: {len(prompts)}")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p[:50]}...")

    results = {}

    # Task 1: Single image generation
    log("TASK 1: Single Image Generation")
    results["single"] = generate_single(
        prompt="A dragon flying over misty mountains, traditional Chinese painting style",
        output_path=os.path.join(BATCH_OUTPUT, "single_dragon.png"),
        seed=42,
    )

    # Task 2: Batch from test_prompts.txt
    log("TASK 2: Batch Generation (test_prompts.txt)")
    results["test_batch"] = len(generate_batch(test_prompts, os.path.join(BATCH_OUTPUT, "test_batch"), start_seed=100)) == len(test_prompts)

    # Task 3: Batch from prompts.txt
    log("TASK 3: Batch Generation (prompts.txt)")
    results["main_batch"] = len(generate_batch(prompts, os.path.join(BATCH_OUTPUT, "main_batch"), start_seed=200)) == len(prompts)

    # Task 4: Series - 山灵
    log("TASK 4: Series Creation - 山灵")
    results["series_shanling"] = generate_series(
        name="山灵",
        theme="Chinese mountain spirits and fairies",
        prompts=prompts,
        output_dir=SERIES_OUTPUT,
        start_seed=300,
    )

    # Task 5: Series - 龙传说
    log("TASK 5: Series Creation - 龙传说")
    results["series_dragon"] = generate_series(
        name="龙传说",
        theme="Dragon legends in Chinese mythology",
        prompts=test_prompts,
        output_dir=SERIES_OUTPUT,
        start_seed=400,
    )

    # Task 6: Single with reference
    log("TASK 6: Single with Reference Style")
    results["reference"] = generate_single(
        prompt="a warrior standing on a cliff edge, sword in hand",
        output_path=os.path.join(BATCH_OUTPUT, "warrior_reference.png"),
        seed=500,
    )

    # Summary
    log("GENERATION COMPLETE")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Batch output: {BATCH_OUTPUT}")
    print(f"Series output: {SERIES_OUTPUT}")

    # Count generated images
    total = 0
    valid = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.endswith(".png"):
                total += 1
                path = os.path.join(root, f)
                from PIL import Image
                import numpy as np
                img = Image.open(path)
                arr = np.array(img)
                if arr.mean() > 10 and arr.std() > 10:
                    valid += 1

    print(f"\nTotal images: {total}")
    print(f"Valid images: {valid}")
    print(f"Blank images: {total - valid}")

    # Task results
    print(f"\nTask Results:")
    for task, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {task}")

    if valid == total and total > 0:
        print(f"\nAll {total} images generated successfully!")
    else:
        print(f"\nSome images may be blank. Check output directory.")


if __name__ == "__main__":
    main()
