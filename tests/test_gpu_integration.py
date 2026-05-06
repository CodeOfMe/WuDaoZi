"""WuDaoZi GPU integration test - all tests run on AMD 7900XTX (ROCm).

Outputs images to test_output/ directory.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wudaozi.api import (
    ToolResult,
    batch_generate,
    create_series,
    generate_image,
    load_reference,
)
from wudaozi.core import WuDaoZiEngine
from wudaozi.tools import TOOLS, dispatch

OUTPUT_DIR = Path(__file__).parent / "test_output"
PROMPTS_FILE = OUTPUT_DIR / "test_prompts.txt"

PROMPTS = [
    "A koi fish leaping from a tranquil lotus pond at golden hour",
    "A traditional Chinese pavilion nestled among cherry blossoms at dawn",
    "A samurai warrior standing on a cliff overlooking the sea at sunset",
]

VAE_OFFLOAD = True


def _clear_gpu():
    import gc
    from wudaozi.api import _engine_cache
    _engine_cache.clear()
    torch.cuda.empty_cache()
    gc.collect()


def setup():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "batch").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "series").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "resolutions").mkdir(parents=True, exist_ok=True)
    PROMPTS_FILE.write_text("\n".join(PROMPTS), encoding="utf-8")


def check_gpu():
    assert torch.cuda.is_available(), "CUDA/ROCm not available!"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {name} ({vram:.1f} GB VRAM)")
    assert "7900" in name or vram > 20, f"Expected 7900 XTX, got {name}"


def test_single_generate():
    print("\n=== Test 1: Single generate_image ===")
    t0 = time.time()
    result = generate_image(
        prompt="A serene Chinese mountain landscape with misty peaks and a waterfall, ink wash painting",
        output=str(OUTPUT_DIR / "test1_single.png"),
        height=768,
        width=1024,
        num_inference_steps=8,
        seed=42,
        vae_offload=VAE_OFFLOAD,
    )
    elapsed = time.time() - t0
    assert result.success, f"Failed: {result.error}"
    img_path = Path(result.data["path"])
    assert img_path.exists(), f"Image not found: {img_path}"
    from PIL import Image
    img = Image.open(img_path)
    print(f"  OK: {img.size} in {elapsed:.1f}s -> {img_path}")


def test_batch_generate():
    print("\n=== Test 2: Batch generate (3 images) ===")
    t0 = time.time()
    result = batch_generate(
        prompts_file=str(PROMPTS_FILE),
        output_dir=str(OUTPUT_DIR / "batch"),
        height=768,
        width=1024,
        num_inference_steps=8,
        start_seed=100,
        vae_offload=VAE_OFFLOAD,
    )
    elapsed = time.time() - t0
    assert result.success, f"Failed: {result.error}"
    assert result.data["count"] == 3
    for p in result.data["paths"]:
        assert Path(p).exists(), f"Image not found: {p}"
    print(f"  OK: {result.data['count']} images in {elapsed:.1f}s")
    for p in result.data["paths"]:
        print(f"    -> {p}")


def test_create_series():
    print("\n=== Test 3: Create series with manifest ===")
    t0 = time.time()
    result = create_series(
        name="FourSeasons",
        theme="Chinese landscape paintings of the four seasons",
        prompts_file=str(PROMPTS_FILE),
        output_dir=str(OUTPUT_DIR / "series"),
        height=768,
        width=1024,
        num_inference_steps=8,
        start_seed=200,
        vae_offload=VAE_OFFLOAD,
        reference_style="traditional Chinese ink wash painting, elegant brushwork",
    )
    elapsed = time.time() - t0
    assert result.success, f"Failed: {result.error}"
    manifest_path = Path(result.data["manifest"])
    assert manifest_path.exists(), f"Manifest not found: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["works"]) == 3
    for w in manifest["works"]:
        assert Path(w["path"]).exists(), f"Image not found: {w['path']}"
    print(f"  OK: {len(manifest['works'])} works, manifest in {elapsed:.1f}s")
    for w in manifest["works"]:
        print(f"    -> {w['path']}")


def test_multi_resolution():
    print("\n=== Test 4: Multi-resolution generation ===")
    from PIL import Image

    for h, w, label in [(1024, 1024, "1024x1024"), (768, 1024, "768x1024"), (512, 512, "512x512")]:
        t0 = time.time()
        result = generate_image(
            prompt="A tranquil bamboo forest with morning light filtering through, Japanese zen garden",
            output=str(OUTPUT_DIR / "resolutions" / f"test_res_{label}.png"),
            height=h,
            width=w,
            num_inference_steps=8,
            seed=777,
            vae_offload=VAE_OFFLOAD,
        )
        elapsed = time.time() - t0
        assert result.success, f"Failed at {label}: {result.error}"
        img = Image.open(result.data["path"])
        print(f"  {label}: {img.size} in {elapsed:.1f}s -> {result.data['path']}")


def test_tools_dispatch():
    print("\n=== Test 5: Tools dispatch ===")
    t0 = time.time()
    result = dispatch("wudaozi_generate_image", {
        "prompt": "A white crane standing gracefully in a lotus pond, Chinese watercolor",
        "output": str(OUTPUT_DIR / "test5_tools_dispatch.png"),
        "height": 768,
        "width": 1024,
        "num_inference_steps": 8,
        "seed": 999,
        "vae_offload": True,
        "reference_style": "traditional Chinese ink painting",
    })
    elapsed = time.time() - t0
    assert result["success"], f"Failed: {result['error']}"
    assert Path(result["data"]["path"]).exists()
    print(f"  OK: dispatch generate in {elapsed:.1f}s -> {result['data']['path']}")

    result2 = dispatch("wudaozi_load_reference", {
        "name": "TestRef",
        "style": "ink wash painting",
        "character": "a scholar in white robes",
    })
    assert result2["success"]
    print(f"  OK: dispatch load_reference -> {result2['data']['name']}")


def test_model_manager():
    print("\n=== Test 6: ModelManager (subprocess) ===")
    import subprocess
    script = '''
import sys, os, time
sys.path.insert(0, os.path.join(os.environ.get("WUDAOZI_MODEL_PATH", ""), "..", "src"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from wudaozi.model_manager import MultiModelEngine
from PIL import Image

t0 = time.time()
engine = MultiModelEngine(model_key="z-image-turbo")
engine.load()
img = engine.generate(
    "A crane flying over the Great Wall at sunset",
    height=768, width=1024, num_inference_steps=8, seed=888,
)
elapsed = time.time() - t0
out_path = "''' + str(OUTPUT_DIR / "test6_model_manager.png") + '''"
img.save(out_path)
print(f"OK: {img.size} in {elapsed:.1f}s -> {out_path}")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
        env={**os.environ, "WUDAOZI_MODEL_PATH": "/home/fred/Documents/GitHub/Others/Z-Image/Z-Image-Turbo"},
        timeout=300,
    )
    out_path = OUTPUT_DIR / "test6_model_manager.png"
    if result.returncode == 0 and out_path.exists():
        img = Image.open(out_path)
        print(f"  OK: {img.size} -> {out_path}")
    else:
        print(f"  FAIL: {result.stderr[-500:] if result.stderr else 'unknown error'}")
        raise AssertionError(f"ModelManager test failed: {result.stderr[-200:] if result.stderr else 'see above'}")


def test_load_reference():
    print("\n=== Test 7: load_reference API ===")
    r1 = load_reference(
        name="DragonStyle",
        style="Chinese dragon motif, gold and red palette",
        character="a warrior in ornate armor with dragon motifs",
    )
    assert r1.success
    print(f"  OK: created reference '{r1.data['name']}'")

    ref_data = {
        "name": "MuseRef",
        "style": "oil painting with impressionist brushstrokes",
        "character": "a woman in white dress holding a parasol",
        "reference_images": ["ref1.jpg"],
    }
    ref_file = OUTPUT_DIR / "test7_ref.json"
    ref_file.write_text(json.dumps(ref_data))
    r2 = load_reference(name="test", from_file=str(ref_file))
    assert r2.success
    assert r2.data["name"] == "MuseRef"
    print(f"  OK: loaded from JSON '{r2.data['name']}'")


def test_engine_direct():
    print("\n=== Test 8: WuDaoZiEngine direct ===")
    _clear_gpu()
    t0 = time.time()
    engine = WuDaoZiEngine(vae_offload=True)
    engine.load()
    img = engine.generate(
        "A magnificent phoenix soaring through clouds, Chinese silk embroidery style",
        height=768,
        width=1024,
        seed=1234,
    )
    elapsed = time.time() - t0
    out_path = OUTPUT_DIR / "test8_engine_direct.png"
    img.save(str(out_path))
    assert out_path.exists()
    print(f"  OK: {img.size} in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    setup()
    check_gpu()

    results = []
    tests = [
        test_single_generate,
        test_batch_generate,
        test_create_series,
        test_multi_resolution,
        test_tools_dispatch,
        test_model_manager,
        test_load_reference,
        test_engine_direct,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
            results.append((test_fn.__name__, "PASS"))
        except Exception as e:
            failed += 1
            results.append((test_fn.__name__, f"FAIL: {e}"))
            print(f"  FAIL: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")
    print(f"Output directory: {OUTPUT_DIR}")

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    (OUTPUT_DIR / "test_report.json").write_text(
        json.dumps(report, default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sys.exit(0 if failed == 0 else 1)