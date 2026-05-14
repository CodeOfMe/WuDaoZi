"""Model manager for downloading and managing multiple quantized image generation models.

Supports downloading from ModelScope (preferred for China users) and HuggingFace (fallback).
Optimized for ROCm/AMD GPUs (7900 XTX etc.) with unified interface.
"""

import gc
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import torch
from loguru import logger

_DEFAULT_SOURCE = "modelscope"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

_MODELS_ROOT = Path(os.environ.get("WUDAOZI_MODELS_DIR", str(_PACKAGE_ROOT / "models")))


def _is_rocm() -> bool:
    """Detect if running on ROCm/AMD GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        return hasattr(torch.version, "hip") and torch.version.hip is not None
    except Exception:
        return False


def _detect_device() -> str:
    """Detect best available device, with ROCm awareness."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm
        return str(xm.xla_device())
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _get_compute_dtype(device: str, model_type: str = "") -> torch.dtype:
    """Get optimal compute dtype for device and model type."""
    if device == "cpu":
        return torch.float32
    if _is_rocm():
        return torch.float16
    if model_type in ("flux", "zimage", "kolors"):
        return torch.bfloat16
    return torch.float16


def _get_gpu_vram_gb() -> float:
    """Get GPU VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        return total / (1024 ** 3)
    except Exception:
        return 0.0


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
    modelscope_id: str = ""
    hf_id: str = ""


AVAILABLE_MODELS = {
    "z-image-turbo": ModelInfo(
        name="Z-Image-Turbo",
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        model_type="zimage",
        description="Z-Image Turbo - Fast Chinese-style image generation",
        default_steps=8,
        supports_guidance=False,
        size_gb=12.0,
        modelscope_id="Tongyi-MAI/Z-Image-Turbo",
        hf_id="Tongyi-MAI/Z-Image-Turbo",
    ),
    "flux-schnell": ModelInfo(
        name="FLUX.1-schnell",
        repo_id="black-forest-labs/FLUX.1-schnell",
        model_type="flux",
        description="FLUX.1 schnell - 12B parameter, 1-4 steps",
        default_steps=4,
        supports_guidance=False,
        size_gb=24.0,
        modelscope_id="AI-ModelScope/FLUX.1-schnell",
        hf_id="black-forest-labs/FLUX.1-schnell",
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
        modelscope_id="AI-ModelScope/FLUX.1-schnell",
        hf_id="strangezoo/flux-schnell-4bit",
    ),
    "sdxl-turbo": ModelInfo(
        name="SDXL-Turbo",
        repo_id="stabilityai/sdxl-turbo",
        model_type="sdxl",
        description="SDXL-Turbo - Adversarial Diffusion Distillation, 1-4 steps",
        default_steps=1,
        supports_guidance=False,
        size_gb=6.5,
        modelscope_id="stabilityai/sdxl-turbo",
        hf_id="stabilityai/sdxl-turbo",
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
        modelscope_id="stabilityai/sdxl-turbo",
        hf_id="stabilityai/sdxl-turbo",
    ),
    "playground-v2": ModelInfo(
        name="Playground-v2-1024px",
        repo_id="playgroundai/playground-v2-1024px",
        model_type="sdxl",
        description="Playground v2 - 1024px, human preference model",
        default_steps=20,
        supports_guidance=True,
        size_gb=5.5,
        modelscope_id="",
        hf_id="playgroundai/playground-v2-1024px",
    ),
    "pixart-alpha": ModelInfo(
        name="PixArt-Alpha",
        repo_id="PixArt-alpha/PixArt-XL-2-1024-MS",
        model_type="pixart",
        description="PixArt-Alpha - 1024px, fast generation",
        default_steps=20,
        supports_guidance=True,
        size_gb=2.4,
        modelscope_id="AI-ModelScope/PixArt-alpha",
        hf_id="PixArt-alpha/PixArt-XL-2-1024-MS",
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
        modelscope_id="AI-ModelScope/PixArt-alpha",
        hf_id="PixArt-alpha/PixArt-XL-2-1024-MS-4bit",
    ),
    "kolors": ModelInfo(
        name="Kolors",
        repo_id="Kwai-Kolors/Kolors",
        model_type="kolors",
        description="Kolors - Kwai fast image generation",
        default_steps=20,
        supports_guidance=True,
        size_gb=13.0,
        modelscope_id="Kwai-Kolors/Kolors",
        hf_id="Kwai-Kolors/Kolors",
    ),
    "aura-flow": ModelInfo(
        name="Aura-Flow",
        repo_id="fal/AuraFlow",
        model_type="flow",
        description="Aura-Flow - Flow matching model",
        default_steps=20,
        supports_guidance=True,
        size_gb=12.0,
        modelscope_id="AI-ModelScope/AuraFlow",
        hf_id="fal/AuraFlow",
    ),
}


def get_model_dir(model_key: str) -> Path:
    """Get the local directory for a model under <package>/models/."""
    return _MODELS_ROOT / model_key


def list_models() -> list[ModelInfo]:
    """List all available models."""
    return list(AVAILABLE_MODELS.values())


def get_model_info(model_key: str) -> Optional[ModelInfo]:
    """Get information about a specific model."""
    return AVAILABLE_MODELS.get(model_key)


def _get_download_source() -> str:
    """Determine download source: modelscope or huggingface.

    Priority: WUDAOZI_DOWNLOAD_SOURCE env var > defaults to modelscope.
    """
    source = os.environ.get("WUDAOZI_DOWNLOAD_SOURCE", "").strip().lower()
    if source in ("huggingface", "hf"):
        return "huggingface"
    return "modelscope"


def download_model(model_key: str, force: bool = False, source: str = "") -> Path:
    """Download a model from ModelScope (preferred) or HuggingFace to models/ directory.

    Args:
        model_key: Key of the model to download
        force: Force re-download even if exists
        source: Download source override ('modelscope' or 'huggingface').
                Empty string uses default (modelscope).

    Returns:
        Path to the downloaded model
    """
    model_info = AVAILABLE_MODELS.get(model_key)
    if not model_info:
        raise ValueError(f"Unknown model: {model_key}")

    target_dir = get_model_dir(model_key)

    if target_dir.exists() and not force:
        if any(target_dir.iterdir()):
            logger.info(f"Model already exists at {target_dir}")
            return target_dir

    source = source or _get_download_source()
    target_dir.mkdir(parents=True, exist_ok=True)

    if source == "huggingface":
        return _download_from_huggingface(model_info, target_dir)
    else:
        if not model_info.modelscope_id:
            raise ValueError(
                f"No ModelScope ID for {model_info.name}. "
                f"Use --source huggingface or set WUDAOZI_DOWNLOAD_SOURCE=huggingface"
            )
        return _download_from_modelscope(model_info, target_dir)


_IGNORE_PATTERNS = [
    "*.onnx",
    "*.onnx_data",
    "*/model.onnx*",
    "*.msgpack",
]


def _download_from_modelscope(model_info: ModelInfo, target_dir: Path) -> Path:
    """Download model from ModelScope to models/ directory.

    Only downloads safetensors and config files, skipping ONNX/bin/msgpack.
    """
    from modelscope import snapshot_download as ms_snapshot_download

    ms_id = model_info.modelscope_id
    if not ms_id:
        raise ValueError(f"No ModelScope ID for {model_info.name}, use HuggingFace instead")

    cache_dir = str(_MODELS_ROOT / ".cache" / "modelscope")
    logger.info(f"Downloading {model_info.name} from ModelScope ({ms_id})...")
    logger.info(f"  Target: {target_dir}")
    logger.info(f"  Cache: {cache_dir}")
    try:
        ms_snapshot_download(
            model_id=ms_id,
            local_dir=str(target_dir),
            cache_dir=cache_dir,
            ignore_file_pattern=_IGNORE_PATTERNS,
        )
        logger.success(f"Downloaded {model_info.name} from ModelScope to {target_dir}")
    except Exception as e:
        # Fallback: try without ignore pattern for older modelscope versions
        logger.warning(f"Ignore pattern not supported, retrying without filter: {e}")
        try:
            ms_snapshot_download(
                model_id=ms_id,
                local_dir=str(target_dir),
                cache_dir=cache_dir,
            )
            logger.success(f"Downloaded {model_info.name} from ModelScope to {target_dir}")
        except Exception as e2:
            logger.error(f"ModelScope download failed: {e2}")
            raise

    return target_dir


def _download_from_huggingface(model_info: ModelInfo, target_dir: Path) -> Path:
    """Download model from HuggingFace to models/ directory.

    Only downloads safetensors and config files, skipping ONNX/bin/msgpack.
    """
    from huggingface_hub import snapshot_download

    hf_id = model_info.hf_id or model_info.repo_id
    cache_dir = str(_MODELS_ROOT / ".cache" / "huggingface")
    logger.info(f"Downloading {model_info.name} from HuggingFace ({hf_id})...")
    logger.info(f"  Target: {target_dir}")
    logger.info(f"  Cache: {cache_dir}")

    try:
        snapshot_download(
            repo_id=hf_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            cache_dir=cache_dir,
            ignore_patterns=_IGNORE_PATTERNS,
        )
        logger.success(f"Downloaded {model_info.name} from HuggingFace to {target_dir}")
    except Exception as e:
        logger.error(f"HuggingFace download failed: {e}")
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
    """Engine that supports multiple image generation models with ROCm/AMD GPU support.

    Unified interface for all model types. Automatically detects ROCm and configures
    optimal dtype and memory settings for AMD GPUs like 7900 XTX.
    """

    def __init__(self, model_key: str = "z-image-turbo", device: Optional[str] = None):
        self.model_key = model_key
        self.model_info = AVAILABLE_MODELS.get(model_key)
        if not self.model_info:
            raise ValueError(f"Unknown model: {model_key}")

        self.device = device or _detect_device()
        self.dtype = _get_compute_dtype(self.device, self.model_info.model_type)
        self.is_rocm = _is_rocm()
        self.gpu_vram_gb = _get_gpu_vram_gb()
        self._pipe = None
        self._local_path: Optional[Path] = None

        if self.is_rocm:
            logger.info(f"ROCm detected: {torch.cuda.get_device_name(0)}, "
                        f"VRAM: {self.gpu_vram_gb:.1f} GB, dtype: {self.dtype}")

    def load(self, low_vram: bool = False) -> None:
        """Load the model with automatic ROCm optimization."""
        if self._pipe is not None:
            return

        model_type = self.model_info.model_type
        logger.info(f"Loading {self.model_info.name} (type={model_type}) on {self.device}, "
                    f"dtype={self.dtype}, rocm={self.is_rocm}")

        if model_type == "flux":
            self._load_flux(low_vram)
        elif model_type == "sdxl":
            self._load_sdxl(low_vram)
        elif model_type == "pixart":
            self._load_pixart(low_vram)
        elif model_type == "kolors":
            self._load_kolors(low_vram)
        elif model_type in ("zimage", "flow"):
            self._load_diffusers_pipeline(low_vram)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _optimize_for_rocm(self) -> None:
        """Apply ROCm-specific optimizations for AMD GPUs."""
        if not self.is_rocm:
            return
        os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("ROCm optimizations applied: expandable_segments enabled")

    def _offload_model(self, low_vram: bool) -> None:
        """Apply memory offloading based on VRAM and low_vram setting."""
        if low_vram or self.gpu_vram_gb < 16:
            self._pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload (low VRAM mode)")
        else:
            self._pipe.to(self.device)
        self._optimize_for_rocm()

    def _load_flux(self, low_vram: bool = False) -> None:
        """Load FLUX.1 model using diffusers."""
        from diffusers import FluxPipeline

        model_path = download_model(self.model_key)
        self._local_path = model_path

        self._pipe = FluxPipeline.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self._offload_model(low_vram)
        logger.info(f"Loaded {self.model_info.name} on {self.device} (dtype={self.dtype})")

    def _load_sdxl(self, low_vram: bool = False) -> None:
        """Load SDXL-Turbo model using diffusers."""
        from diffusers import AutoPipelineForText2Image

        model_path = download_model(self.model_key)
        self._local_path = model_path

        variant = "fp16" if self.device != "cpu" else None
        load_kwargs = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True}
        if variant and self.model_info.quantization != "fp16":
            load_kwargs["variant"] = variant

        self._pipe = AutoPipelineForText2Image.from_pretrained(
            str(model_path), **load_kwargs,
        )
        self._offload_model(low_vram)
        logger.info(f"Loaded {self.model_info.name} on {self.device} (dtype={self.dtype})")

    def _load_pixart(self, low_vram: bool = False) -> None:
        """Load PixArt-Alpha model using diffusers."""
        from diffusers import PixArtAlphaPipeline

        model_path = download_model(self.model_key)
        self._local_path = model_path

        self._pipe = PixArtAlphaPipeline.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self._offload_model(low_vram)
        logger.info(f"Loaded {self.model_info.name} on {self.device} (dtype={self.dtype})")

    def _load_kolors(self, low_vram: bool = False) -> None:
        """Load Kolors model using diffusers."""
        from diffusers import KolorsPipeline

        model_path = download_model(self.model_key)
        self._local_path = model_path

        self._pipe = KolorsPipeline.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self._offload_model(low_vram)
        logger.info(f"Loaded {self.model_info.name} on {self.device} (dtype={self.dtype})")

    def _load_diffusers_pipeline(self, low_vram: bool = False) -> None:
        """Load Z-Image or other flow models using diffusers pipeline."""
        from diffusers import AutoPipelineForText2Image

        model_path = download_model(self.model_key)
        self._local_path = model_path

        self._pipe = AutoPipelineForText2Image.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self._offload_model(low_vram)
        logger.info(f"Loaded {self.model_info.name} on {self.device} (dtype={self.dtype})")

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
        """Generate an image using the unified interface."""
        if self._pipe is None:
            self.load()

        num_inference_steps = num_inference_steps or self.model_info.default_steps
        logger.info(f"Generating on {self.device} (rocm={self.is_rocm}, "
                    f"dtype={self.dtype}, steps={num_inference_steps})")

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
        """Generate using diffusers pipeline with ROCm memory management."""
        torch.cuda.empty_cache()
        gc.collect()

        generator_device = self.device
        if self.is_rocm and hasattr(self._pipe, "device"):
            generator_device = self._pipe.device
        generator = torch.Generator(generator_device).manual_seed(seed)

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
    """List all downloaded models in models/ directory."""
    downloaded = []
    for model_key, info in AVAILABLE_MODELS.items():
        model_dir = get_model_dir(model_key)
        if model_dir.exists() and any(model_dir.iterdir()):
            downloaded.append(model_key)
    for model_key in list(AVAILABLE_MODELS.keys()):
        model_dir = get_model_dir(model_key)
        if (model_dir / "model_index.json").exists() and model_key not in downloaded:
            downloaded.append(model_key)
    return downloaded


def is_downloaded(model_key: str) -> bool:
    """Check if a specific model is downloaded to models/ directory."""
    model_dir = get_model_dir(model_key)
    return model_dir.exists() and any(model_dir.iterdir())


def _resolve_model_path_for_size(model_key: str) -> Optional[Path]:
    """Resolve the actual on-disk path for size calculation."""
    model_dir = get_model_dir(model_key)
    if model_dir.exists() and any(model_dir.iterdir()):
        return model_dir
    return None


def get_model_size(model_key: str) -> dict:
    """Get the size of a model on disk."""
    model_path = _resolve_model_path_for_size(model_key)
    if model_path is None:
        return {"exists": False, "size_gb": 0.0, "path": ""}

    total_size = 0
    for path in model_path.rglob("*"):
        if path.is_file():
            total_size += path.stat().st_size

    return {
        "exists": True,
        "size_gb": total_size / (1024**3),
        "path": str(model_path),
    }


if __name__ == "__main__":
    print("Available models:")
    for info in list_models():
        size_info = get_model_size(info.name.lower().replace(".", "-").replace(" ", "-"))
        status = f"({size_info['size_gb']:.1f} GB)" if size_info["exists"] else "(not downloaded)"
        print(f"  - {info.name}: {info.description} {status}")
