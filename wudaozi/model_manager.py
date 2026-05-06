"""Model manager for downloading and managing multiple quantized image generation models."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import torch
from loguru import logger


class ModelProtocol(Protocol):
    """Protocol for model pipelines."""

    def __call__(self, prompt: str, **kwargs):
        ...


@dataclass
class ModelInfo:
    """Information about a model."""

    name: str
    repo_id: str
    model_type: str
    description: str
    default_steps: int = 4
    supports_guidance: bool = True
    quantization: Optional[str] = None
    size_gb: float = 0.0
    local_path: Optional[str] = None


AVAILABLE_MODELS = {
    "z-image-turbo": ModelInfo(
        name="Z-Image-Turbo",
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        model_type="zimage",
        description="Z-Image Turbo - Fast Chinese-style image generation",
        default_steps=8,
        supports_guidance=False,
        size_gb=12.0,
    ),
    "flux-schnell": ModelInfo(
        name="FLUX.1-schnell",
        repo_id="black-forest-labs/FLUX.1-schnell",
        model_type="flux",
        description="FLUX.1 schnell - 12B parameter, 1-4 steps",
        default_steps=4,
        supports_guidance=False,
        size_gb=24.0,
    ),
    "flux-schnell-quantized": ModelInfo(
        name="FLUX.1-schnell-4bit",
        repo_id="strangezoo/flux-schnell-4bit",
        model_type="flux",
        description="FLUX.1 schnell - 4-bit quantized, smaller size",
        default_steps=4,
        supports_guidance=False,
        quantization="4bit",
        size_gb=7.0,
    ),
    "sdxl-turbo": ModelInfo(
        name="SDXL-Turbo",
        repo_id="stabilityai/sdxl-turbo",
        model_type="sdxl",
        description="SDXL-Turbo - Adversarial Diffusion Distillation, 1-4 steps",
        default_steps=1,
        supports_guidance=False,
        size_gb=6.5,
    ),
    "sdxl-turbo-fp16": ModelInfo(
        name="SDXL-Turbo-FP16",
        repo_id="stabilityai/sdxl-turbo",
        model_type="sdxl",
        description="SDXL-Turbo - FP16 variant",
        default_steps=1,
        supports_guidance=False,
        quantization="fp16",
        size_gb=6.5,
    ),
    "playground-v2": ModelInfo(
        name="Playground-v2-1024px",
        repo_id="playgroundai/playground-v2-1024px",
        model_type="sdxl",
        description="Playground v2 - 1024px, human preference model",
        default_steps=20,
        supports_guidance=True,
        size_gb=5.5,
    ),
    "pixart-alpha": ModelInfo(
        name="PixArt-Alpha",
        repo_id="PixArt-alpha/PixArt-XL-2-1024-MS",
        model_type="pixart",
        description="PixArt-Alpha - 1024px, fast generation",
        default_steps=20,
        supports_guidance=True,
        size_gb=2.4,
    ),
    "pixart-alpha-quantized": ModelInfo(
        name="PixArt-Alpha-4bit",
        repo_id="PixArt-alpha/PixArt-XL-2-1024-MS-4bit",
        model_type="pixart",
        description="PixArt-Alpha - 4-bit quantized, ~2.2GB",
        default_steps=20,
        supports_guidance=True,
        quantization="4bit",
        size_gb=2.2,
    ),
    "kolors": ModelInfo(
        name="Kolors",
        repo_id="Kwai-Kolors/Kolors",
        model_type="kolors",
        description="Kolors - Kwai fast image generation",
        default_steps=20,
        supports_guidance=True,
        size_gb=13.0,
    ),
    "aura-flow": ModelInfo(
        name="Aura-Flow",
        repo_id="fal/AuraFlow",
        model_type="flow",
        description="Aura-Flow - Flow matching model",
        default_steps=20,
        supports_guidance=True,
        size_gb=12.0,
    ),
}


def get_model_dir(model_key: str) -> Path:
    """Get the local directory for a model."""
    cache_dir = Path.home() / ".cache" / "wudaozi" / "models"
    return cache_dir / model_key


def list_models() -> list[ModelInfo]:
    """List all available models."""
    return list(AVAILABLE_MODELS.values())


def get_model_info(model_key: str) -> Optional[ModelInfo]:
    """Get information about a specific model."""
    return AVAILABLE_MODELS.get(model_key)


def download_model(model_key: str, force: bool = False) -> Path:
    """Download a model from HuggingFace.

    Args:
        model_key: Key of the model to download
        force: Force re-download even if exists

    Returns:
        Path to the downloaded model
    """
    from huggingface_hub import snapshot_download

    model_info = AVAILABLE_MODELS.get(model_key)
    if not model_info:
        raise ValueError(f"Unknown model: {model_key}")

    target_dir = get_model_dir(model_key)

    if target_dir.exists() and not force:
        if any(target_dir.iterdir()):
            logger.info(f"Model already exists at {target_dir}")
            return target_dir

    logger.info(f"Downloading {model_info.name} from {model_info.repo_id}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=model_info.repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.success(f"Downloaded {model_info.name} to {target_dir}")
    except Exception as e:
        logger.error(f"Failed to download {model_info.name}: {e}")
        raise

    return target_dir


def download_quantized_model(
    base_model: str,
    quantization: str = "4bit",
    force: bool = False,
) -> Path:
    """Download a quantized version of a model.

    Args:
        base_model: Base model key (e.g., 'flux-schnell')
        quantization: Quantization type ('4bit', '8bit')
        force: Force re-download

    Returns:
        Path to the quantized model
    """
    quant_key = f"{base_model}-{quantization}"

    if quant_key in AVAILABLE_MODELS:
        return download_model(quant_key, force)

    logger.info(f"Creating {quantization} quantization of {base_model}...")
    base_path = download_model(base_model, force=False)

    return base_path


class MultiModelEngine:
    """Engine that supports multiple image generation models."""

    def __init__(self, model_key: str = "z-image-turbo", device: Optional[str] = None):
        self.model_key = model_key
        self.model_info = AVAILABLE_MODELS.get(model_key)
        if not self.model_info:
            raise ValueError(f"Unknown model: {model_key}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None
        self._local_path: Optional[Path] = None

    def load(self, low_vram: bool = False) -> None:
        """Load the model."""
        if self._pipe is not None:
            return

        model_type = self.model_info.model_type

        if model_type == "flux":
            self._load_flux(low_vram)
        elif model_type == "sdxl":
            self._load_sdxl(low_vram)
        elif model_type == "pixart":
            self._load_pixart(low_vram)
        elif model_type == "kolors":
            self._load_kolors(low_vram)
        elif model_type == "zimage":
            self._load_zimage()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _load_flux(self, low_vram: bool = False) -> None:
        """Load FLUX.1 model using diffusers."""
        from diffusers import FluxPipeline

        model_path = download_model(self.model_key)
        self._local_path = model_path

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self._pipe = FluxPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        if low_vram:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(self.device)

        self._pipe.eval()
        logger.info(f"Loaded FLUX.1-schnell on {self.device}")

    def _load_sdxl(self, low_vram: bool = False) -> None:
        """Load SDXL-Turbo model using diffusers."""
        from diffusers import AutoPipelineForText2Image

        model_path = download_model(self.model_key)
        self._local_path = model_path

        variant = "fp16" if self.device == "cuda" else None
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._pipe = AutoPipelineForText2Image.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            variant=variant,
            low_cpu_mem_usage=True,
        )

        if low_vram:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(self.device)

        self._pipe.eval()
        logger.info(f"Loaded SDXL-Turbo on {self.device}")

    def _load_pixart(self, low_vram: bool = False) -> None:
        """Load PixArt-Alpha model using diffusers."""
        from diffusers import AutoPipelineForText2Image

        model_path = download_model(self.model_key)
        self._local_path = model_path

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self._pipe = AutoPipelineForText2Image.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        if low_vram:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(self.device)

        self._pipe.eval()
        logger.info(f"Loaded PixArt-Alpha on {self.device}")

    def _load_kolors(self, low_vram: bool = False) -> None:
        """Load Kolors model using diffusers."""
        from diffusers import AutoPipelineForText2Image

        model_path = download_model(self.model_key)
        self._local_path = model_path

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self._pipe = AutoPipelineForText2Image.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        if low_vram:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(self.device)

        self._pipe.eval()
        logger.info(f"Loaded Kolors on {self.device}")

    def _load_zimage(self) -> None:
        """Load Z-Image model using local loader."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from utils import ensure_model_weights, load_from_local_dir

        model_path = ensure_model_weights(self.model_info.repo_id, verify=False)
        self._local_path = model_path

        self._pipe = load_from_local_dir(
            str(model_path),
            device=self.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        logger.info(f"Loaded Z-Image-Turbo on {self.device}")

    def generate(
        self,
        prompt: str,
        *,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 0.0,
        seed: int = 42,
        **kwargs,
    ):
        """Generate an image."""
        if self._pipe is None:
            self.load()

        num_inference_steps = num_inference_steps or self.model_info.default_steps

        if self.model_info.model_type == "zimage":
            return self._generate_zimage(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        else:
            return self._generate_diffusers(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                **kwargs,
            )

    def _generate_diffusers(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        **kwargs,
    ):
        """Generate using diffusers pipeline."""
        import gc

        torch.cuda.empty_cache()
        gc.collect()

        generator = torch.Generator(self.device).manual_seed(seed)

        if self.model_info.model_type == "sdxl":
            height = min(height, 1024)
            width = min(width, 1024)

        result = self._pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale if self.model_info.supports_guidance else 0.0,
            generator=generator,
            **kwargs,
        )

        torch.cuda.empty_cache()
        gc.collect()

        return result.images[0]

    def _generate_zimage(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        **kwargs,
    ):
        """Generate using Z-Image native pipeline."""
        from zimage import generate as zimage_generate

        generator = torch.Generator(self.device).manual_seed(seed)

        images = zimage_generate(
            prompt=prompt,
            **self._pipe,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return images[0]

    def batch_generate(
        self,
        prompts: list[str],
        *,
        output_dir: str = "outputs",
        start_seed: int = 42,
        **kwargs,
    ) -> list[str]:
        """Generate multiple images."""
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for idx, prompt in enumerate(prompts, start=1):
            seed = start_seed + idx - 1
            img = self.generate(prompt, seed=seed, **kwargs)

            slug = "".join(c.lower() if c.isalnum() else "-" for c in prompt[:60])
            filename = f"{idx:03d}-{slug}.png"
            out_path = output_dir / filename
            img.save(out_path)
            saved_paths.append(str(out_path))

        return saved_paths


def list_downloaded_models() -> list[str]:
    """List all downloaded models."""
    cache_dir = Path.home() / ".cache" / "wudaozi" / "models"
    if not cache_dir.exists():
        return []

    downloaded = []
    for model_key in AVAILABLE_MODELS:
        model_dir = cache_dir / model_key
        if model_dir.exists() and any(model_dir.iterdir()):
            downloaded.append(model_key)

    return downloaded


def get_model_size(model_key: str) -> dict:
    """Get the size of a model on disk."""
    model_dir = get_model_dir(model_key)
    if not model_dir.exists():
        return {"exists": False, "size_gb": 0.0}

    total_size = 0
    for path in model_dir.rglob("*"):
        if path.is_file():
            total_size += path.stat().st_size

    return {
        "exists": True,
        "size_gb": total_size / (1024**3),
        "path": str(model_dir),
    }


if __name__ == "__main__":
    print("Available models:")
    for info in list_models():
        size_info = get_model_size(info.name.lower().replace(".", "-").replace(" ", "-"))
        status = f"({size_info['size_gb']:.1f} GB)" if size_info["exists"] else "(not downloaded)"
        print(f"  - {info.name}: {info.description} {status}")
