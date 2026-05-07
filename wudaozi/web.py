"""WuDaoZi Web UI - Gradio-based painting studio interface."""

import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from . import __version__

_engine_cache = {}


def _get_engine(model_key, vae_offload=True):
    global _engine_cache
    if model_key == "z-image-turbo":
        cache_key = ("zimage", vae_offload)
        if cache_key not in _engine_cache:
            from .core import WuDaoZiEngine
            _engine_cache[cache_key] = WuDaoZiEngine(vae_offload=vae_offload if vae_offload else None)
            _engine_cache[cache_key].load()
        return _engine_cache[cache_key]
    else:
        cache_key = (model_key,)
        if cache_key not in _engine_cache:
            from .model_manager import MultiModelEngine
            _engine_cache[cache_key] = MultiModelEngine(model_key=model_key)
            _engine_cache[cache_key].load()
        return _engine_cache[cache_key]


def _model_steps(model_key):
    from .model_manager import AVAILABLE_MODELS
    info = AVAILABLE_MODELS.get(model_key)
    return info.default_steps if info else 8


def _model_supports_guidance(model_key):
    from .model_manager import AVAILABLE_MODELS
    info = AVAILABLE_MODELS.get(model_key)
    return info.supports_guidance if info else False


def _is_downloaded(model_key):
    from .model_manager import is_downloaded
    return is_downloaded(model_key)


def generate_single(prompt, height, width, steps, seed, guidance, style, character, model_key, vae_offload):
    prompt = (prompt or "").strip()
    style = (style or "").strip()
    character = (character or "").strip()
    if not prompt:
        return None, "", "Error: enter a prompt"

    if not _is_downloaded(model_key):
        return None, None, f"Model '{model_key}' not downloaded. Go to Models tab to download first."

    t0 = time.time()
    try:
        engine = _get_engine(model_key=model_key, vae_offload=vae_offload)
        from .core import ReferenceProfile

        effective_prompt = prompt
        if style or character:
            ref = ReferenceProfile(name="web", style_description=style, character_description=character)
            effective_prompt = ref.to_prompt_prefix() + prompt

        gen_kwargs = dict(
            prompt=effective_prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            seed=int(seed),
        )
        if _model_supports_guidance(model_key):
            gen_kwargs["guidance_scale"] = float(guidance)
        else:
            gen_kwargs["guidance_scale"] = 0.0

        img = engine.generate(**gen_kwargs)

        out_dir = Path(tempfile.mkdtemp(prefix="wudaozi_web_"))
        out_path = str(out_dir / "output.png")
        img.save(out_path)

        elapsed = time.time() - t0
        info = f"Done in {elapsed:.1f}s | {int(width)}\u00d7{int(height)} | {steps} steps | seed {int(seed)} | model {model_key}"
        if style or character:
            info += f"\nstyle: {style or 'N/A'} | char: {character or 'N/A'}"
        return out_path, out_path, info
    except Exception as e:
        import traceback
        return None, None, f"Error: {e}\n{traceback.format_exc()}"


def generate_series(name, theme, prompts_text, height, width, steps, seed, guidance,
                    style, character, model_key, vae_offload):
    name = (name or "").strip() or "series"
    theme = (theme or "").strip()
    prompts_text = (prompts_text or "").strip()
    style = (style or "").strip()
    character = (character or "").strip()
    prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]
    if not prompts:
        return [], None, "Error: enter at least one prompt"

    if not _is_downloaded(model_key):
        return [], None, f"Model '{model_key}' not downloaded. Go to Models tab to download first."

    t0 = time.time()
    try:
        engine = _get_engine(model_key=model_key, vae_offload=vae_offload)
        from .core import GenerationConfig, ReferenceProfile, SeriesConfig

        cfg_kwargs = dict(
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            seed=int(seed),
        )
        if _model_supports_guidance(model_key):
            cfg_kwargs["guidance_scale"] = float(guidance)
        else:
            cfg_kwargs["guidance_scale"] = 0.0
        config = GenerationConfig(**cfg_kwargs)

        ref = None
        if style or character:
            ref = ReferenceProfile(name=name, style_description=style, character_description=character)

        out_dir = tempfile.mkdtemp(prefix="wudaozi_series_")
        series_cfg = SeriesConfig(
            name=name,
            theme=theme or name,
            prompts=prompts,
            reference=ref,
            config=config,
            output_dir=out_dir,
            start_seed=int(seed),
        )
        result = engine.create_series(series_cfg)

        elapsed = time.time() - t0
        paths = [w["path"] for w in result["works"]]
        gallery = [(p, Path(p).stem) for p in paths]

        series_dir = result.get("output_dir", out_dir)
        zip_path = _zip_series(series_dir, name)
        zip_size_mb = Path(zip_path).stat().st_size / 1024 / 1024

        info = f"Generated {len(paths)} images in {elapsed:.1f}s | model {model_key}\n"
        info += f"Series: {name} | Theme: {theme or name}\n"
        if style or character:
            info += f"Shared context - style: {style or 'N/A'} | character: {character or 'N/A'}\n"
        info += f"ZIP: {zip_size_mb:.1f} MB"
        return gallery, zip_path, info
    except Exception as e:
        import traceback
        return [], None, f"Error: {e}\n{traceback.format_exc()}"


def _zip_series(series_dir, name):
    zip_path = tempfile.mktemp(suffix=".zip", prefix=f"wudaozi_{slugify(name)}_")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(series_dir):
            for f in sorted(files):
                filepath = Path(root) / f
                arcname = str(Path(name) / filepath.relative_to(series_dir))
                zf.write(filepath, arcname)
    return zip_path


def slugify(text, max_len=40):
    result = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    result = "-".join(p for p in result.split("-") if p)
    return result[:max_len].rstrip("-") or "output"


def download_model_fn(model_key):
    if not model_key:
        return "Please select a model to download.", _refresh_models_fn()
    from .model_manager import download_model, AVAILABLE_MODELS
    info = AVAILABLE_MODELS.get(model_key)
    if not info:
        return f"Unknown model: {model_key}", _refresh_models_fn()
    try:
        path = download_model(model_key)
        size_mb = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024 / 1024
        return f"Downloaded {info.name} ({size_mb:.0f} MB) to {path}", _refresh_models_fn()
    except Exception as e:
        return f"Download failed: {e}", _refresh_models_fn()


def _refresh_models_fn():
    from .model_manager import AVAILABLE_MODELS, is_downloaded, get_model_size
    lines = []
    for key, info in AVAILABLE_MODELS.items():
        downloaded = is_downloaded(key)
        size_info = get_model_size(key)
        status = f"downloaded ({size_info['size_gb']:.1f} GB)" if downloaded else "not downloaded"
        lines.append(f"- **{info.name}** (`{key}`): {info.description} [{status}]")
    return "\n".join(lines)


def build_app():
    import gradio as gr

    gpu_name = "CPU"
    vram_info = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            vram_info = f" ({vram:.0f} GB)"
    except Exception:
        pass

    from .model_manager import AVAILABLE_MODELS

    model_choices = list(AVAILABLE_MODELS.keys())

    with gr.Blocks(title=f"WuDaoZi \u5434\u9053\u5b50 v{__version__}") as app:
        gr.Markdown(
            f"# \U0001f3a8 WuDaoZi \u5434\u9053\u5b50 v{__version__}\n"
            f"Batch painting studio powered by Z-Image | **{gpu_name}{vram_info}**"
        )

        with gr.Row():
            model_key = gr.Dropdown(
                choices=model_choices,
                value="z-image-turbo",
                label="Model",
                scale=3,
                allow_custom_value=False,
            )
            vae_offload = gr.Checkbox(
                value=True,
                label="VAE Offload (<30GB VRAM)",
                scale=2,
            )

        with gr.Tabs():
            with gr.Tab("\U0001f3a8 Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the image you want to paint...",
                            lines=4,
                        )
                        with gr.Row():
                            height = gr.Slider(256, 2048, value=1024, step=128, label="Height")
                            width = gr.Slider(256, 2048, value=1024, step=128, label="Width")
                        with gr.Row():
                            steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                            seed = gr.Number(value=42, label="Seed", precision=0)
                        guidance = gr.Slider(0.0, 20.0, value=0.0, step=0.5, label="Guidance Scale (CFG)", visible=False)
                        gen_btn = gr.Button("\u2728 Generate", variant="primary", size="lg")
                    with gr.Column(scale=3):
                        output_image = gr.Image(label="Output", type="filepath")
                        gen_download = gr.File(label="\U0001f4e5 Download Image")
                        gen_info = gr.Textbox(label="Info", interactive=False)

                def _update_steps_guidance(mk):
                    info = AVAILABLE_MODELS.get(mk)
                    ds = info.default_steps if info else 8
                    sg = info.supports_guidance if info else False
                    return gr.update(value=ds), gr.update(visible=sg)

                model_key.change(fn=_update_steps_guidance, inputs=[model_key], outputs=[steps, guidance])

                gen_btn.click(
                    fn=generate_single,
                    inputs=[prompt, height, width, steps, seed, guidance,
                            gr.Textbox(visible=False), gr.Textbox(visible=False),
                            model_key, vae_offload],
                    outputs=[output_image, gen_download, gen_info],
                )

            with gr.Tab("\U0001f5bc Reference Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ref_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the image...",
                            lines=3,
                        )
                        gr.Markdown("### \U0001f3ab Shared Context (applied to every image)")
                        with gr.Row():
                            ref_style = gr.Textbox(
                                label="Style",
                                placeholder="ink wash painting, traditional Chinese style",
                            )
                            ref_character = gr.Textbox(
                                label="Character",
                                placeholder="a warrior in red armor, flowing cape",
                            )
                        with gr.Row():
                            ref_height = gr.Slider(256, 2048, value=1024, step=128, label="Height")
                            ref_width = gr.Slider(256, 2048, value=1024, step=128, label="Width")
                        with gr.Row():
                            ref_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                            ref_seed = gr.Number(value=42, label="Seed", precision=0)
                        ref_guidance = gr.Slider(0.0, 20.0, value=0.0, step=0.5, label="Guidance Scale (CFG)", visible=False)
                        ref_gen_btn = gr.Button("\u2728 Generate with Reference", variant="primary")
                    with gr.Column(scale=3):
                        ref_output_image = gr.Image(label="Output", type="filepath")
                        ref_download = gr.File(label="\U0001f4e5 Download Image")
                        ref_gen_info = gr.Textbox(label="Info", interactive=False)

                model_key.change(fn=_update_steps_guidance, inputs=[model_key], outputs=[ref_steps, ref_guidance])

                ref_gen_btn.click(
                    fn=generate_single,
                    inputs=[ref_prompt, ref_height, ref_width, ref_steps, ref_seed, ref_guidance,
                            ref_style, ref_character, model_key, vae_offload],
                    outputs=[ref_output_image, ref_download, ref_gen_info],
                )

            with gr.Tab("\U0001f4da Series Generate"):
                gr.Markdown(
                    "**Series mode**: generate multiple images sharing the same style, character, and theme.\n"
                    "The shared context (style + character) is prepended to every prompt, ensuring visual consistency."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            series_name = gr.Textbox(label="Series Name", placeholder="e.g. Four Seasons, Heroes")
                            series_theme = gr.Textbox(label="Theme", placeholder="e.g. Chinese landscape in four seasons")
                        gr.Markdown("### \U0001f3ab Shared Context (applied to ALL images)")
                        with gr.Row():
                            series_style = gr.Textbox(label="Style", placeholder="ink wash painting, traditional Chinese style")
                            series_character = gr.Textbox(label="Character", placeholder="a wandering swordsman in white robe")
                        series_prompts = gr.Textbox(
                            label="Prompts (one per line)",
                            placeholder="standing on a mountain peak at sunrise\nmeditating by a waterfall\nwalking through bamboo forest",
                            lines=6,
                        )
                        with gr.Row():
                            series_height = gr.Slider(256, 2048, value=1024, step=128, label="Height")
                            series_width = gr.Slider(256, 2048, value=1024, step=128, label="Width")
                        with gr.Row():
                            series_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                            series_seed = gr.Number(value=42, label="Start Seed", precision=0)
                        series_guidance = gr.Slider(0.0, 20.0, value=0.0, step=0.5, label="Guidance Scale", visible=False)
                        series_btn = gr.Button("\U0001f4da Generate Series", variant="primary", size="lg")
                    with gr.Column(scale=3):
                        series_gallery = gr.Gallery(label="Results", columns=3, height=500, format="png")
                        series_zip = gr.File(label="\U0001f4e5 Download All (ZIP)")
                        series_info = gr.Textbox(label="Info", interactive=False, lines=4)

                model_key.change(fn=_update_steps_guidance, inputs=[model_key], outputs=[series_steps, series_guidance])

                series_btn.click(
                    fn=generate_series,
                    inputs=[series_name, series_theme, series_prompts,
                            series_height, series_width, series_steps, series_seed, series_guidance,
                            series_style, series_character, model_key, vae_offload],
                    outputs=[series_gallery, series_zip, series_info],
                )

            with gr.Tab("\U0001f4cb Reference Profile"):
                gr.Markdown("Create a reusable reference profile JSON for style/character consistency.")
                with gr.Column():
                    ref_name = gr.Textbox(label="Name", placeholder="e.g. Hero, Landscape")
                    ref_pf_style = gr.Textbox(label="Style", placeholder="ink wash painting, traditional Chinese style")
                    ref_pf_char = gr.Textbox(label="Character", placeholder="a warrior in red armor, flowing cape")
                    save_ref_btn = gr.Button("\U0001f4be Save Reference Profile")
                    ref_file_output = gr.File(label="\U0001f4e5 Download JSON")
                    ref_status = gr.Textbox(label="Status", interactive=False)

                save_ref_btn.click(
                    fn=save_reference,
                    inputs=[ref_name, ref_pf_style, ref_pf_char],
                    outputs=[ref_file_output, ref_status],
                )

            with gr.Tab("\U0001f4e6 Models"):
                gr.Markdown("### Download & manage models\nModels are stored in `./models/` directory.")
                models_status = gr.Markdown(_refresh_models_fn())
                with gr.Row():
                    dl_model = gr.Dropdown(
                        choices=model_choices,
                        value="flux-schnell",
                        label="Select model to download",
                    )
                    dl_btn = gr.Button("\U0001f4e5 Download", variant="primary")
                dl_result = gr.Textbox(label="Download Status", interactive=False, lines=3)
                refresh_btn = gr.Button("\U0001f504 Refresh List")

                dl_btn.click(fn=download_model_fn, inputs=[dl_model], outputs=[dl_result, models_status])
                refresh_btn.click(fn=lambda: _refresh_models_fn(), outputs=[models_status])

    return app


def save_reference(name, style, character):
    name = (name or "").strip()
    style = (style or "").strip()
    character = (character or "").strip()
    if not name:
        return None, "Error: name is required"
    ref_data = {
        "name": name,
        "style": style,
        "character": character,
        "reference_images": [],
    }
    out_path = tempfile.mktemp(suffix=".json", prefix="wudaozi_ref_")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ref_data, f, ensure_ascii=False, indent=2)
    return out_path, f"Reference profile saved: {name}"


def run_web(host="0.0.0.0", port=7860, share=False):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is required for web UI. Install with: pip install wudaozi[web]", file=sys.stderr)
        sys.exit(1)

    app = build_app()
    print(f"WuDaoZi Web UI starting on http://{host}:{port}")
    app.launch(server_name=host, server_port=port, share=share, theme=gr.themes.Soft())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WuDaoZi Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    args = parser.parse_args()
    run_web(host=args.host, port=args.port, share=args.share)