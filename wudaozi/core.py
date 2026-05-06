"""Core engine for WuDaoZi painting studio."""

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import torch  # noqa: E402

ZIMAGE_SRC = str(Path(__file__).resolve().parent.parent.parent / "src")

if ZIMAGE_SRC not in sys.path:
    sys.path.insert(0, ZIMAGE_SRC)

from utils import (  # noqa: E402
    AttentionBackend,
    ensure_model_weights,
    load_from_local_dir,
    set_attention_backend,
)
from zimage import generate as zimage_generate  # noqa: E402


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm

        return str(xm.xla_device())
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def slugify(text: str, max_len: int = 60) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:max_len].rstrip("-") or "prompt"


@dataclass
class ReferenceProfile:
    name: str
    style_description: str = ""
    character_description: str = ""
    reference_image_paths: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_prompt_prefix(self) -> str:
        parts = []
        if self.character_description:
            parts.append(self.character_description)
        if self.style_description:
            parts.append(self.style_description)
        return ", ".join(parts) + ", " if parts else ""


@dataclass
class GenerationConfig:
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 8
    guidance_scale: float = 0.0
    seed: int = 42
    dtype: str = "bfloat16"
    compile_model: bool = False
    attention_backend: str = "_native_flash"
    low_vram: bool = False


@dataclass
class SeriesConfig:
    name: str
    theme: str
    prompts: list[str] = field(default_factory=list)
    reference: Optional[ReferenceProfile] = None
    config: GenerationConfig = field(default_factory=GenerationConfig)
    output_dir: str = "outputs"
    start_seed: int = 42


def resolve_model_path(explicit: str = "") -> str:
    """Resolve model path with priority: explicit > env > config > auto > default.

    Priority order:
      1. Explicit path passed by caller (CLI --model / API model_path)
      2. WUDAOZI_MODEL_PATH environment variable
      3. ~/.wudaozi/config.json → {"model_path": "..."}
      4. Auto-detect: Z-Image-Turbo/ directory next to WuDaoZi/
      5. Fallback: ckpts/Z-Image-Turbo (relative, may trigger HuggingFace download)
    """
    if explicit:
        return explicit
    env_path = os.environ.get("WUDAOZI_MODEL_PATH", "")
    if env_path:
        return env_path
    config_path = Path.home() / ".wudaozi" / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if cfg.get("model_path"):
                return cfg["model_path"]
        except (json.JSONDecodeError, OSError):
            pass
    sibling = Path(__file__).resolve().parent.parent.parent / "Z-Image-Turbo"
    if sibling.is_dir() and (sibling / "model_index.json").exists():
        return str(sibling)
    return "ckpts/Z-Image-Turbo"


class WuDaoZiEngine:
    def __init__(
        self,
        model_path: str = "",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
        attention_backend: Optional[str] = None,
        low_vram: bool = False,
    ):
        self.model_path = resolve_model_path(model_path)
        self.device = device or select_device()
        self.dtype = dtype or torch.bfloat16
        self.compile_model = compile_model
        self.attention_backend = attention_backend or os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
        self.low_vram = low_vram or bool(int(os.environ.get("WUDAOZI_LOW_VRAM", "0")))
        self._components = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        resolved_path = ensure_model_weights(self.model_path, verify=False)
        if self.low_vram and self.device == "cuda":
            self._components = self._load_low_vram(resolved_path)
        else:
            self._components = load_from_local_dir(
                resolved_path, device=self.device, dtype=self.dtype, compile=self.compile_model
            )
        AttentionBackend.print_available_backends()
        set_attention_backend(self.attention_backend)
        self._loaded = True

    def _load_low_vram(self, model_path: str) -> dict:
        """Use diffusers ZImagePipeline with bitsandbytes 8-bit quantization for low VRAM.

        Strategy for RTX 4060 8GB:
        - Use diffusers ZImagePipeline with bitsandbytes_8bit quantization
        - Move pipeline to CPU first, then enable sequential CPU offload
        - Quantized model fits in 8GB VRAM with proper offloading
        - Cache pipeline to avoid reloading and memory fragmentation
        """
        import gc
        import os
        from pathlib import Path
        from loguru import logger
        from diffusers import ZImagePipeline
        from diffusers.quantizers import PipelineQuantizationConfig

        # Check if pipeline is already loaded
        if hasattr(self, "_pipe") and self._pipe is not None:
            logger.info("Reusing cached quantized pipeline")
            return {"pipe": self._pipe}

        # Enable expandable segments for better memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Low VRAM mode: diffusers ZImagePipeline with bitsandbytes 8-bit quantization")
        self.device = "cuda"
        self.attention_backend = "native"
        resolved = ensure_model_weights(model_path, verify=False)

        # Configure 8-bit quantization for transformer
        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=["transformer"],
        )

        # Load pipeline with quantization
        logger.info("Loading ZImagePipeline with bitsandbytes_8bit quantization...")
        pipe = ZImagePipeline.from_pretrained(
            resolved,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=quant_config,
        )

        # Move to CPU first to avoid GPU OOM during offload setup
        logger.info("Moving pipeline to CPU...")
        pipe.to("cpu")

        # Enable sequential CPU offload for memory efficiency
        logger.info("Enabling sequential CPU offload...")
        pipe.enable_sequential_cpu_offload()

        # Cache the pipeline
        self._pipe = pipe

        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Low VRAM setup complete: quantized pipeline with sequential CPU offload")
        return {"pipe": pipe}

    @property
    def components(self) -> dict:
        if not self._loaded:
            self.load()
        return self._components

    def generate(
        self,
        prompt: str,
        *,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 8,
        guidance_scale: float = 0.0,
        seed: int = 42,
        negative_prompt: Optional[str] = None,
    ):  # noqa: ANN204
        import gc

        comp = self.components

        # Check if using diffusers pipeline (low_vram mode)
        if "pipe" in comp:
            pipe = comp["pipe"]

            # Clear GPU memory before generation
            torch.cuda.empty_cache()
            gc.collect()

            generator = torch.Generator("cpu").manual_seed(seed)

            # Use latent output type to avoid VAE OOM, then decode on CPU
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent",
            )

            # Move latents to CPU for VAE decode
            latents = result.images.to("cpu")

            # Decode VAE on CPU
            pipe.vae.to("cpu")
            with torch.no_grad():
                # Scale latents for VAE
                latents = latents / pipe.vae.config.scaling_factor + getattr(pipe.vae.config, "shift_factor", 0.0)
                image = pipe.vae.decode(latents, return_dict=False)[0]

            # Move VAE back to CPU (it should already be there)
            pipe.vae.to("cpu")

            # Convert to PIL image
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            from PIL import Image
            image = Image.fromarray(image[0])

            # Clear GPU memory after generation
            torch.cuda.empty_cache()
            gc.collect()

            return image

        # Original zimage generation
        generator = torch.Generator(self.device).manual_seed(seed)
        kwargs = dict(
            prompt=prompt,
            **comp,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if negative_prompt and guidance_scale > 1.0:
            kwargs["negative_prompt"] = negative_prompt
        images = zimage_generate(**kwargs)
        return images[0]

    def generate_with_reference(
        self,
        prompt: str,
        reference: ReferenceProfile,
        **kwargs,
    ):  # noqa: ANN204
        full_prompt = reference.to_prompt_prefix() + prompt
        return self.generate(full_prompt, **kwargs)

    def batch_generate(
        self,
        prompts: list[str],
        *,
        output_dir: str = "outputs",
        reference: Optional[ReferenceProfile] = None,
        config: Optional[GenerationConfig] = None,
        start_seed: int = 42,
    ) -> list[str]:
        if config is None:
            config = GenerationConfig()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for idx, prompt in enumerate(prompts, start=1):
            seed = start_seed + idx - 1
            if reference:
                img = self.generate_with_reference(
                    prompt,
                    reference,
                    height=config.height,
                    width=config.width,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    seed=seed,
                )
            else:
                img = self.generate(
                    prompt,
                    height=config.height,
                    width=config.width,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    seed=seed,
                )
            filename = f"{idx:03d}-{slugify(prompt)}.png"
            out_path = output_dir / filename
            img.save(out_path)
            saved_paths.append(str(out_path))
        return saved_paths

    def create_series(
        self,
        series_config: SeriesConfig,
    ) -> dict:
        output_dir = Path(series_config.output_dir) / slugify(series_config.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        series_meta = {
            "name": series_config.name,
            "theme": series_config.theme,
            "reference": None,
            "config": {
                "height": series_config.config.height,
                "width": series_config.config.width,
                "num_inference_steps": series_config.config.num_inference_steps,
                "guidance_scale": series_config.config.guidance_scale,
                "start_seed": series_config.start_seed,
            },
            "works": [],
        }
        if series_config.reference:
            series_meta["reference"] = {
                "name": series_config.reference.name,
                "style": series_config.reference.style_description,
                "character": series_config.reference.character_description,
                "reference_images": series_config.reference.reference_image_paths,
            }
        paths = self.batch_generate(
            series_config.prompts,
            output_dir=output_dir,
            reference=series_config.reference,
            config=series_config.config,
            start_seed=series_config.start_seed,
        )
        for i, p in enumerate(paths, start=1):
            series_meta["works"].append({"index": i, "path": p, "prompt": series_config.prompts[i - 1]})
        meta_path = output_dir / "series.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(series_meta, f, ensure_ascii=False, indent=2)
        series_meta["output_dir"] = str(output_dir)
        series_meta["manifest"] = str(meta_path)
        return series_meta


def load_reference_profile(
    name: str,
    style: str = "",
    character: str = "",
    reference_images: Optional[list[str]] = None,
    from_file: Optional[str] = None,
) -> ReferenceProfile:
    if from_file:
        path = Path(from_file)
        if not path.exists():
            raise FileNotFoundError(f"Reference profile file not found: {from_file}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ReferenceProfile(
            name=data.get("name", name),
            style_description=data.get("style", style),
            character_description=data.get("character", character),
            reference_image_paths=data.get("reference_images", reference_images or []),
            metadata=data.get("metadata", {}),
        )
    return ReferenceProfile(
        name=name,
        style_description=style,
        character_description=character,
        reference_image_paths=reference_images or [],
    )


def read_prompts(path: str) -> list[str]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts
