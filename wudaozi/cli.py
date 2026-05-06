"""WuDaoZi CLI - Command-line interface for batch painting studio."""

import argparse
import json
import logging
import os
import pathlib
import sys

from . import __version__


def setup_logging(verbose: bool, quiet: bool):
    level = logging.INFO
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wudaozi",
        description=(
            "WuDaoZi - Batch painting studio powered by Z-Image (named after the greatest painter in Chinese history)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wudaozi generate --prompt "A dragon flying over mountains"
  wudaozi batch --prompts prompts.txt --output-dir outputs/
  wudaozi series --name "Summer Landscape" --theme "Chinese ink painting" --prompts prompts.txt
  wudaozi reference --name "Hero" --character "A warrior in red armor" --style "ink wash painting"
  wudaozi config --model-path /path/to/Z-Image-Turbo
  wudaozi config --show
  wudaozi models --list
  wudaozi models --download flux-schnell
  wudaozi gui

Model path resolution priority:
  CLI --model > env WUDAOZI_MODEL_PATH > ~/.wudaozi/config.json > auto-detect > ckpts/Z-Image-Turbo

Available models:
  z-image-turbo    - Z-Image Turbo (default)
  flux-schnell     - FLUX.1 schnell (12B, 1-4 steps)
  sdxl-turbo       - SDXL-Turbo (fast, 1-4 steps)
  pixart-alpha     - PixArt-Alpha (2.4GB, fast)
  kolors           - Kolors (Kwai fast model)
""",
    )
    parser.add_argument("-V", "--version", action="version", version=f"wudaozi {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output path")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output results as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate a single image")
    gen_parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    gen_parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    gen_parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    gen_parser.add_argument("--steps", type=int, default=8, help="Inference steps (default: 8)")
    gen_parser.add_argument("--guidance", type=float, default=0.0, help="Guidance scale (default: 0.0)")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    gen_parser.add_argument("--model", default="", help="Model path (env: WUDAOZI_MODEL_PATH, default: auto-detect)")
    gen_parser.add_argument(
        "--model-key",
        default="z-image-turbo",
        help="Model key (z-image-turbo, flux-schnell, sdxl-turbo, etc.)",
    )
    gen_parser.add_argument("--low-vram", action="store_true", help="Low VRAM mode (VAE on CPU, decode on GPU)")
    gen_parser.add_argument("--reference-style", default="", help="Style reference description")
    gen_parser.add_argument("--reference-character", default="", help="Character reference description")

    batch_parser = subparsers.add_parser("batch", help="Batch generate from prompts file")
    batch_parser.add_argument("--prompts", required=True, help="Path to prompts file (one per line)")
    batch_parser.add_argument("--output-dir", default="outputs", help="Output directory")
    batch_parser.add_argument("--height", type=int, default=1024)
    batch_parser.add_argument("--width", type=int, default=1024)
    batch_parser.add_argument("--steps", type=int, default=8)
    batch_parser.add_argument("--guidance", type=float, default=0.0)
    batch_parser.add_argument("--start-seed", type=int, default=42)
    batch_parser.add_argument("--model", default="", help="Model path (env: WUDAOZI_MODEL_PATH, default: auto-detect)")
    batch_parser.add_argument(
        "--model-key",
        default="z-image-turbo",
        help="Model key (z-image-turbo, flux-schnell, sdxl-turbo, etc.)",
    )
    batch_parser.add_argument("--low-vram", action="store_true", help="Low VRAM mode")
    batch_parser.add_argument("--reference-style", default="")
    batch_parser.add_argument("--reference-character", default="")
    batch_parser.add_argument("--reference-file", default="")

    series_parser = subparsers.add_parser("series", help="Create a themed artwork series")
    series_parser.add_argument("--name", required=True, help="Series name")
    series_parser.add_argument("--theme", required=True, help="Series theme description")
    series_parser.add_argument("--prompts", required=True, help="Path to prompts file")
    series_parser.add_argument("--output-dir", default="outputs")
    series_parser.add_argument("--height", type=int, default=1024)
    series_parser.add_argument("--width", type=int, default=1024)
    series_parser.add_argument("--steps", type=int, default=8)
    series_parser.add_argument("--guidance", type=float, default=0.0)
    series_parser.add_argument("--start-seed", type=int, default=42)
    series_parser.add_argument("--model", default="", help="Model path (env: WUDAOZI_MODEL_PATH, default: auto-detect)")
    series_parser.add_argument(
        "--model-key",
        default="z-image-turbo",
        help="Model key (z-image-turbo, flux-schnell, sdxl-turbo, etc.)",
    )
    series_parser.add_argument("--low-vram", action="store_true", help="Low VRAM mode")
    series_parser.add_argument("--reference-style", default="")
    series_parser.add_argument("--reference-character", default="")
    series_parser.add_argument("--reference-file", default="")

    ref_parser = subparsers.add_parser("reference", help="Create a reference profile JSON")
    ref_parser.add_argument("--name", required=True, help="Reference profile name")
    ref_parser.add_argument("--style", default="", help="Style description")
    ref_parser.add_argument("--character", default="", help="Character description")
    ref_parser.add_argument("--reference-images", nargs="*", default=[], help="Reference image paths")

    config_parser = subparsers.add_parser("config", help="View or set persistent configuration")
    config_parser.add_argument("--model-path", default="", help="Set model path (saved to ~/.wudaozi/config.json)")
    config_parser.add_argument(
        "--model-key",
        default="",
        help="Set default model key (saved to ~/.wudaozi/config.json)",
    )
    config_parser.add_argument("--show", action="store_true", help="Show current resolved configuration")

    models_parser = subparsers.add_parser("models", help="List and manage available models")
    models_parser.add_argument("--list", action="store_true", help="List all available models")
    models_parser.add_argument(
        "--download",
        default="",
        help="Download a specific model (e.g., flux-schnell, sdxl-turbo)",
    )
    models_parser.add_argument("--downloaded", action="store_true", help="Show only downloaded models")

    subparsers.add_parser("gui", help="Launch GUI studio")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose, args.quiet)

    try:
        if args.command == "generate":
            _cmd_generate(args)
        elif args.command == "batch":
            _cmd_batch(args)
        elif args.command == "series":
            _cmd_series(args)
        elif args.command == "reference":
            _cmd_reference(args)
        elif args.command == "config":
            _cmd_config(args)
        elif args.command == "models":
            _cmd_models(args)
        elif args.command == "gui":
            _cmd_gui(args)
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _output(result, args):
    if args.json_output:
        print(json.dumps(result.to_dict(), ensure_ascii=False))
    else:
        if result.success:
            print(json.dumps(result.data, ensure_ascii=False, indent=2))
        else:
            print(f"Error: {result.error}", file=sys.stderr)


def _cmd_generate(args):
    from .api import generate_image

    output_path = str(args.output) if args.output else "output.png"
    result = generate_image(
        prompt=args.prompt,
        output=output_path,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        model_path=args.model,
        model_key=args.model_key,
        low_vram=args.low_vram,
        reference_style=args.reference_style,
        reference_character=args.reference_character,
    )
    _output(result, args)


def _cmd_batch(args):
    from .api import batch_generate

    result = batch_generate(
        prompts_file=args.prompts,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        start_seed=args.start_seed,
        model_path=args.model,
        model_key=args.model_key,
        low_vram=args.low_vram,
        reference_style=args.reference_style,
        reference_character=args.reference_character,
        reference_file=args.reference_file,
    )
    _output(result, args)


def _cmd_series(args):
    from .api import create_series

    result = create_series(
        name=args.name,
        theme=args.theme,
        prompts_file=args.prompts,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        start_seed=args.start_seed,
        model_path=args.model,
        model_key=args.model_key,
        low_vram=args.low_vram,
        reference_style=args.reference_style,
        reference_character=args.reference_character,
        reference_file=args.reference_file,
    )
    _output(result, args)


def _cmd_reference(args):
    from .api import load_reference

    result = load_reference(
        name=args.name,
        style=args.style,
        character=args.character,
        reference_images=args.reference_images,
    )
    _output(result, args)


def _cmd_config(args):
    from .core import resolve_model_path

    config_path = pathlib.Path.home() / ".wudaozi" / "config.json"
    if args.model_path or args.model_key:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if config_path.exists():
            import json as _json

            existing = _json.loads(config_path.read_text(encoding="utf-8"))
        if args.model_path:
            existing["model_path"] = args.model_path
            print(f"Config saved to {config_path}")
            print(f"  model_path = {args.model_path}")
        if args.model_key:
            existing["model_key"] = args.model_key
            print(f"Config saved to {config_path}")
            print(f"  model_key = {args.model_key}")
        config_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        resolved = resolve_model_path()
        print(f"  resolved   = {resolved}")
    elif args.show or not args.model_path:
        resolved = resolve_model_path()
        env_path = os.environ.get("WUDAOZI_MODEL_PATH", "(not set)")
        env_model_key = os.environ.get("WUDAOZI_MODEL_KEY", "(not set)")
        print("Model path resolution:")
        print("  --model arg:    (not provided)")
        print(f"  WUDAOZI_MODEL_PATH:  {env_path}")
        print(f"  WUDAOZI_MODEL_KEY:  {env_model_key}")
        print(f"  config file:    {config_path}")
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            print(f"    model_path = {cfg.get('model_path', '(not set)')}")
            print(f"    model_key = {cfg.get('model_key', '(not set)')}")
        else:
            print("    (file does not exist)")
        print(f"  resolved path: {resolved}")


def _cmd_models(args):
    from .model_manager import (
        AVAILABLE_MODELS,
        download_model,
        get_model_info,
        get_model_size,
        list_downloaded_models,
    )

    if args.download:
        model_key = args.download
        if model_key not in AVAILABLE_MODELS:
            print(f"Unknown model: {model_key}")
            print("Available models:")
            for key in AVAILABLE_MODELS:
                print(f"  - {key}")
            sys.exit(1)
        print(f"Downloading {model_key}...")
        path = download_model(model_key)
        size = get_model_size(model_key)
        print(f"Downloaded to: {path}")
        print(f"Size: {size['size_gb']:.2f} GB")
    elif args.downloaded:
        downloaded = list_downloaded_models()
        if not downloaded:
            print("No models downloaded yet.")
            print("Use 'wudaozi models --download <model-key>' to download a model.")
        else:
            print("Downloaded models:")
            for key in downloaded:
                size = get_model_size(key)
                info = get_model_info(key)
                print(f"  - {info.name if info else key}: {size['size_gb']:.2f} GB")
    elif args.list:
        print("Available models:")
        for key, info in AVAILABLE_MODELS.items():
            size = get_model_size(key)
            status = f"{size['size_gb']:.1f} GB" if size["exists"] else "not downloaded"
            print(f"  - {info.name}")
            print(f"    Key: {key}")
            print(f"    Type: {info.model_type}")
            print(f"    Size: ~{info.size_gb:.1f} GB")
            print(f"    Status: {status}")
            print(f"    Description: {info.description}")
            print()
    else:
        downloaded = list_downloaded_models()
        print("WuDaoZi Model Manager")
        print("=" * 50)
        print(f"Downloaded models: {len(downloaded)}")
        print()
        print("Commands:")
        print("  wudaozi models --list                    List all available models")
        print("  wudaozi models --download <model-key>   Download a model")
        print("  wudaozi models --downloaded             Show downloaded models")
        print()
        print("Available models:")
        for key, info in AVAILABLE_MODELS.items():
            size = get_model_size(key)
            status = "downloaded" if size["exists"] else "not downloaded"
            print(f"  {key}: {info.name} ({status})")


def _cmd_gui(args):
    from .gui import run_gui

    run_gui()
