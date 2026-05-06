"""WuDaoZi Web UI - Gradio-based painting studio interface."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from . import __version__

_engine = None


def _get_or_create_engine(model_key="z-image-turbo", vae_offload=True):
    global _engine
    if _engine is None:
        from .core import WuDaoZiEngine
        _engine = WuDaoZiEngine(vae_offload=vae_offload if vae_offload else None)
        _engine.load()
    return _engine


def generate_single(prompt, height, width, steps, seed, style, character, model_key, vae_offload):
    prompt = (prompt or "").strip()
    style = (style or "").strip()
    character = (character or "").strip()
    if not prompt:
        return None, "Error: prompt cannot be empty"

    t0 = time.time()
    try:
        engine = _get_or_create_engine(model_key=model_key, vae_offload=vae_offload)
        from .core import ReferenceProfile

        effective_prompt = prompt
        if style or character:
            ref = ReferenceProfile(
                name="web",
                style_description=style,
                character_description=character,
            )
            effective_prompt = ref.to_prompt_prefix() + prompt

        img = engine.generate(
            effective_prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            seed=int(seed),
        )

        out_dir = Path(tempfile.mkdtemp(prefix="wudaozi_web_"))
        out_path = str(out_dir / "output.png")
        img.save(out_path)

        elapsed = time.time() - t0
        info = f"Done in {elapsed:.1f}s | {int(width)}x{int(height)} | {steps} steps | seed {int(seed)}"
        if style or character:
            info += f" | style: {style or 'N/A'} | char: {character or 'N/A'}"
        return out_path, info
    except Exception as e:
        return None, f"Error: {e}"


def generate_batch(prompts_text, height, width, steps, seed, style, character, model_key, vae_offload):
    prompts_text = (prompts_text or "").strip()
    style = (style or "").strip()
    character = (character or "").strip()
    prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]
    if not prompts:
        return [], "Error: no prompts provided"

    t0 = time.time()
    try:
        engine = _get_or_create_engine(model_key=model_key, vae_offload=vae_offload)
        from .core import GenerationConfig, ReferenceProfile

        config = GenerationConfig(
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            seed=int(seed),
        )
        ref = None
        if style or character:
            ref = ReferenceProfile(
                name="batch_web",
                style_description=style,
                character_description=character,
            )

        out_dir = tempfile.mkdtemp(prefix="wudaozi_batch_")
        paths = engine.batch_generate(
            prompts,
            output_dir=out_dir,
            reference=ref,
            config=config,
            start_seed=int(seed),
        )

        elapsed = time.time() - t0
        gallery = [(p, Path(p).stem) for p in paths]
        info = f"Generated {len(paths)} images in {elapsed:.1f}s"
        return gallery, info
    except Exception as e:
        return [], f"Error: {e}"


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


def build_app():
    import gradio as gr

    with gr.Blocks(title=f"WuDaoZi \u5434\u9053\u5b50 v{__version__}") as app:
        gr.Markdown(
            f"# WuDaoZi \u5434\u9053\u5b50 v{__version__}\n"
            "Batch painting studio powered by Z-Image"
        )

        with gr.Row():
            model_key = gr.Dropdown(
                choices=["z-image-turbo"],
                value="z-image-turbo",
                label="Model",
            )
            vae_offload = gr.Checkbox(
                value=True,
                label="VAE Offload (recommended for <30GB VRAM)",
            )

        with gr.Tabs():
            with gr.Tab("Generate"):
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
                        gen_btn = gr.Button("Generate", variant="primary", size="lg")
                    with gr.Column(scale=3):
                        output_image = gr.Image(label="Output", type="filepath")
                        gen_info = gr.Textbox(label="Info", interactive=False)

                gen_btn.click(
                    fn=generate_single,
                    inputs=[prompt, height, width, steps, seed,
                            gr.Textbox(visible=False), gr.Textbox(visible=False),
                            model_key, vae_offload],
                    outputs=[output_image, gen_info],
                )

            with gr.Tab("Reference Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ref_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the image...",
                            lines=3,
                        )
                        with gr.Row():
                            ref_style = gr.Textbox(
                                label="Style",
                                placeholder="ink wash painting, traditional Chinese",
                            )
                            ref_character = gr.Textbox(
                                label="Character",
                                placeholder="a warrior in red armor",
                            )
                        with gr.Row():
                            ref_height = gr.Slider(256, 2048, value=1024, step=128, label="Height")
                            ref_width = gr.Slider(256, 2048, value=1024, step=128, label="Width")
                        with gr.Row():
                            ref_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                            ref_seed = gr.Number(value=42, label="Seed", precision=0)
                        ref_gen_btn = gr.Button("Generate with Reference", variant="primary")
                    with gr.Column(scale=3):
                        ref_output_image = gr.Image(label="Output", type="filepath")
                        ref_gen_info = gr.Textbox(label="Info", interactive=False)

                ref_gen_btn.click(
                    fn=generate_single,
                    inputs=[ref_prompt, ref_height, ref_width, ref_steps, ref_seed,
                            ref_style, ref_character, model_key, vae_offload],
                    outputs=[ref_output_image, ref_gen_info],
                )

            with gr.Tab("Batch Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        batch_prompts = gr.Textbox(
                            label="Prompts (one per line)",
                            placeholder="A koi fish leaping from lotus pond\nA dragon over mountains\n...",
                            lines=8,
                        )
                        with gr.Row():
                            batch_style = gr.Textbox(label="Style (all)", placeholder="optional")
                            batch_character = gr.Textbox(label="Character (all)", placeholder="optional")
                        with gr.Row():
                            batch_height = gr.Slider(256, 2048, value=1024, step=128, label="Height")
                            batch_width = gr.Slider(256, 2048, value=1024, step=128, label="Width")
                        with gr.Row():
                            batch_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                            batch_seed = gr.Number(value=42, label="Start Seed", precision=0)
                        batch_btn = gr.Button("Batch Generate", variant="primary")
                    with gr.Column(scale=3):
                        batch_gallery = gr.Gallery(label="Results", columns=3, height=500)
                        batch_info = gr.Textbox(label="Info", interactive=False)

                batch_btn.click(
                    fn=generate_batch,
                    inputs=[batch_prompts, batch_height, batch_width, batch_steps, batch_seed,
                            batch_style, batch_character, model_key, vae_offload],
                    outputs=[batch_gallery, batch_info],
                )

            with gr.Tab("Reference Profile"):
                with gr.Column():
                    ref_name = gr.Textbox(label="Name", placeholder="e.g. Hero, Landscape")
                    ref_pf_style = gr.Textbox(
                        label="Style", placeholder="ink wash painting, traditional Chinese style"
                    )
                    ref_pf_char = gr.Textbox(
                        label="Character", placeholder="a warrior in red armor, flowing cape"
                    )
                    save_ref_btn = gr.Button("Save Reference Profile")
                    ref_file_output = gr.File(label="Download")
                    ref_status = gr.Textbox(label="Status", interactive=False)

                save_ref_btn.click(
                    fn=save_reference,
                    inputs=[ref_name, ref_pf_style, ref_pf_char],
                    outputs=[ref_file_output, ref_status],
                )

        gpu_name = "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        gr.Markdown(f"**Device:** {gpu_name} | **Version:** {__version__}")

    return app


def run_web(host="0.0.0.0", port=7860, share=False):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is required for web UI. Install with: pip install wudaozi[web]", file=sys.stderr)
        sys.exit(1)

    app = build_app()
    print(f"WuDaoZi Web UI starting on http://{host}:{port}")
    app.launch(server_name=host, server_port=port, share=share, theme=gr.themes.Soft())