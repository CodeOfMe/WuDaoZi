"""WuDaoZi comprehensive integration tests - all generation tasks with CPU offload."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Correct model path
MODEL_PATH = "/home/fred/Documents/GitHub/Others/Z-Image/Z-Image-Turbo"

# Test configuration - optimized for RTX 4060 8GB with CPU offload
TEST_CONFIG = {
    "model_path": MODEL_PATH,
    "low_vram": True,  # CPU offload for low VRAM GPU
    "height": 512,  # Smaller for faster testing
    "width": 512,
    "num_inference_steps": 4,  # Fewer steps for testing
    "guidance_scale": 0.0,
}

# Test prompts
SINGLE_PROMPT = "A beautiful mountain landscape with clouds"

BATCH_PROMPTS = [
    "一条金色巨龙盘旋在云雾缭绕的山峰之间",
    "仙鹤在松枝上栖息，月光如水",
]

SERIER_PROMPTS = [
    "穿白色飘袍的仙女在云雾缭绕的山间抚琴",
    "一位仙人在瀑布旁修炼，周身仙气环绕",
]

REFERENCE_PROFILE = {
    "name": "test_reference",
    "style": "Chinese ink wash painting style",
    "character": "a fairy in white robes",
}


def log(msg: str):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def verify_image(path: str) -> bool:
    """Verify that an image file exists and is valid."""
    from PIL import Image

    if not Path(path).exists():
        print(f"  ERROR: File not found: {path}")
        return False
    try:
        img = Image.open(path)
        img.verify()
        # Re-open after verify (verify closes the file)
        img = Image.open(path)
        print(f"  OK: {path} ({img.size[0]}x{img.size[1]}, {img.mode})")
        return True
    except Exception as e:
        print(f"  ERROR: Invalid image {path}: {e}")
        return False


def test_single_generate(output_dir: str) -> bool:
    """Test 1: Single image generation with CPU offload."""
    log("TEST 1: Single Image Generation (CPU offload)")

    from wudaozi.api import generate_image

    output_path = os.path.join(output_dir, "single_test.png")

    result = generate_image(
        prompt=SINGLE_PROMPT,
        output=output_path,
        height=TEST_CONFIG["height"],
        width=TEST_CONFIG["width"],
        num_inference_steps=TEST_CONFIG["num_inference_steps"],
        guidance_scale=TEST_CONFIG["guidance_scale"],
        seed=42,
        model_path=TEST_CONFIG["model_path"],
        low_vram=TEST_CONFIG["low_vram"],
    )

    if not result.success:
        print(f"  FAILED: {result.error}")
        return False

    print(f"  Generated: {result.data['path']}")
    print(f"  Metadata: {result.metadata}")

    return verify_image(output_path)


def test_batch_generate(output_dir: str) -> bool:
    """Test 2: Batch generation from prompts file with CPU offload."""
    log("TEST 2: Batch Generation (CPU offload)")

    from wudaozi.api import batch_generate

    # Create temporary prompts file
    prompts_file = os.path.join(output_dir, "test_prompts.txt")
    with open(prompts_file, "w", encoding="utf-8") as f:
        for prompt in BATCH_PROMPTS:
            f.write(prompt + "\n")

    batch_output_dir = os.path.join(output_dir, "batch_output")

    result = batch_generate(
        prompts_file=prompts_file,
        output_dir=batch_output_dir,
        height=TEST_CONFIG["height"],
        width=TEST_CONFIG["width"],
        num_inference_steps=TEST_CONFIG["num_inference_steps"],
        guidance_scale=TEST_CONFIG["guidance_scale"],
        start_seed=100,
        model_path=TEST_CONFIG["model_path"],
        low_vram=TEST_CONFIG["low_vram"],
    )

    if not result.success:
        print(f"  FAILED: {result.error}")
        return False

    print(f"  Generated {result.data['count']} images:")
    all_ok = True
    for path in result.data["paths"]:
        if not verify_image(path):
            all_ok = False

    return all_ok


def test_reference_profile(output_dir: str) -> bool:
    """Test 3: Reference profile creation and loading."""
    log("TEST 3: Reference Profile")

    from wudaozi.api import load_reference

    # Create reference profile JSON
    ref_file = os.path.join(output_dir, "reference.json")
    with open(ref_file, "w", encoding="utf-8") as f:
        json.dump(REFERENCE_PROFILE, f, ensure_ascii=False, indent=2)

    # Load reference from file
    result = load_reference(name="test", from_file=ref_file)

    if not result.success:
        print(f"  FAILED: {result.error}")
        return False

    print(f"  Reference loaded: {result.data}")

    # Verify data
    if result.data["name"] != REFERENCE_PROFILE["name"]:
        print(f"  FAILED: Name mismatch")
        return False
    if result.data["style"] != REFERENCE_PROFILE["style"]:
        print(f"  FAILED: Style mismatch")
        return False

    print("  Reference profile test passed")
    return True


def test_generate_with_reference(output_dir: str) -> bool:
    """Test 4: Single generation with reference style/character."""
    log("TEST 4: Generate with Reference (CPU offload)")

    from wudaozi.api import generate_image

    output_path = os.path.join(output_dir, "reference_test.png")

    result = generate_image(
        prompt="standing on a mountain peak",
        output=output_path,
        height=TEST_CONFIG["height"],
        width=TEST_CONFIG["width"],
        num_inference_steps=TEST_CONFIG["num_inference_steps"],
        guidance_scale=TEST_CONFIG["guidance_scale"],
        seed=200,
        model_path=TEST_CONFIG["model_path"],
        low_vram=TEST_CONFIG["low_vram"],
        reference_style=REFERENCE_PROFILE["style"],
        reference_character=REFERENCE_PROFILE["character"],
    )

    if not result.success:
        print(f"  FAILED: {result.error}")
        return False

    print(f"  Generated: {result.data['path']}")
    return verify_image(output_path)


def test_series_creation(output_dir: str) -> bool:
    """Test 5: Series creation with consistent style."""
    log("TEST 5: Series Creation (CPU offload)")

    from wudaozi.api import create_series

    # Create prompts file for series
    prompts_file = os.path.join(output_dir, "series_prompts.txt")
    with open(prompts_file, "w", encoding="utf-8") as f:
        for prompt in SERIER_PROMPTS:
            f.write(prompt + "\n")

    series_output_dir = os.path.join(output_dir, "series_output")

    result = create_series(
        name="test_series",
        theme="Chinese fairy tale",
        prompts_file=prompts_file,
        output_dir=series_output_dir,
        height=TEST_CONFIG["height"],
        width=TEST_CONFIG["width"],
        num_inference_steps=TEST_CONFIG["num_inference_steps"],
        guidance_scale=TEST_CONFIG["guidance_scale"],
        start_seed=300,
        model_path=TEST_CONFIG["model_path"],
        low_vram=TEST_CONFIG["low_vram"],
        reference_style=REFERENCE_PROFILE["style"],
        reference_character=REFERENCE_PROFILE["character"],
    )

    if not result.success:
        print(f"  FAILED: {result.error}")
        return False

    print(f"  Series created: {result.data.get('name', 'unknown')}")
    print(f"  Output dir: {result.data.get('output_dir', 'unknown')}")
    print(f"  Manifest: {result.data.get('manifest', 'unknown')}")
    print(f"  Works: {len(result.data.get('works', []))}")

    # Verify all images
    all_ok = True
    for work in result.data.get("works", []):
        if not verify_image(work["path"]):
            all_ok = False

    # Verify manifest
    manifest_path = result.data.get("manifest")
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"  Manifest valid: {manifest.get('name', 'unknown')}")
    else:
        print(f"  WARNING: Manifest not found")

    return all_ok


def test_tool_dispatch() -> bool:
    """Test 6: OpenAI function-calling tool dispatch."""
    log("TEST 6: Tool Dispatch")

    from wudaozi.tools import TOOLS, dispatch

    # Verify tools schema
    if not isinstance(TOOLS, list) or len(TOOLS) < 4:
        print(f"  FAILED: Expected at least 4 tools, got {len(TOOLS)}")
        return False

    print(f"  Tools available: {[t['function']['name'] for t in TOOLS]}")

    # Test dispatch with reference (no model needed)
    result = dispatch("wudaozi_load_reference", {"name": "dispatch_test", "style": "test style"})
    if not result.get("success"):
        print(f"  FAILED: {result.get('error')}")
        return False

    print(f"  Dispatch test passed: {result['data']['name']}")
    return True


def test_cli_integration() -> bool:
    """Test 7: CLI integration tests."""
    log("TEST 7: CLI Integration")

    import subprocess

    # Test version flag
    result = subprocess.run(
        [sys.executable, "-m", "wudaozi", "-V"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        print(f"  FAILED: Version check failed")
        return False
    print(f"  Version: {result.stdout.strip()}")

    # Test help
    result = subprocess.run(
        [sys.executable, "-m", "wudaozi", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        print(f"  FAILED: Help check failed")
        return False
    if "--json" not in result.stdout:
        print(f"  FAILED: Missing --json flag in help")
        return False
    print(f"  CLI help OK (unified flags present)")

    return True


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("  WuDaoZi Integration Test Suite")
    print("  Model: Z-Image-Turbo")
    print("  Mode: CPU offload (low_vram)")
    print("=" * 60)

    # Verify model path
    model_index = Path(MODEL_PATH) / "model_index.json"
    if not model_index.exists():
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print(f"Expected: {model_index}")
        sys.exit(1)
    print(f"\nModel verified: {MODEL_PATH}")

    # Create temporary output directory
    output_dir = tempfile.mkdtemp(prefix="wudaozi_test_")
    print(f"Output directory: {output_dir}")

    results = {}

    try:
        # Test 1: Tool dispatch (no model needed)
        results["tool_dispatch"] = test_tool_dispatch()

        # Test 2: CLI integration (no model needed)
        results["cli_integration"] = test_cli_integration()

        # Test 3: Reference profile (no model needed)
        results["reference_profile"] = test_reference_profile(output_dir)

        # Test 4: Single generation (needs model)
        results["single_generate"] = test_single_generate(output_dir)

        # Test 5: Batch generation (needs model)
        results["batch_generate"] = test_batch_generate(output_dir)

        # Test 6: Generate with reference (needs model)
        results["generate_with_reference"] = test_generate_with_reference(output_dir)

        # Test 7: Series creation (needs model)
        results["series_creation"] = test_series_creation(output_dir)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Summary
        print("\n" + "=" * 60)
        print("  TEST SUMMARY")
        print("=" * 60)

        passed = 0
        failed = 0
        for name, ok in results.items():
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"  [{status}] {name}")

        print(f"\n  Total: {passed} passed, {failed} failed, {len(results)} total")
        print(f"  Output directory: {output_dir}")

        if failed > 0:
            print(f"\n  Some tests failed! Check output directory for details.")
            sys.exit(1)
        else:
            print(f"\n  All tests passed!")


if __name__ == "__main__":
    main()
