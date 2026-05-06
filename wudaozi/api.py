"""WuDaoZi Python API with ToolResult pattern."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import __version__

# Global engine cache for low_vram mode
_engine_cache = {}


def _get_engine(model_path: str = "", low_vram: bool = False, model_key: str = "z-image-turbo"):
    """Get or create a cached engine instance."""
    if model_key and model_key != "z-image-turbo":
        from .model_manager import MultiModelEngine

        cache_key = (model_key, low_vram)
        if cache_key not in _engine_cache:
            _engine_cache[cache_key] = MultiModelEngine(model_key=model_key)
        return _engine_cache[cache_key]

    cache_key = (model_path, low_vram)
    if cache_key not in _engine_cache:
        from .core import WuDaoZiEngine

        _engine_cache[cache_key] = WuDaoZiEngine(model_path=model_path, low_vram=low_vram)
    return _engine_cache[cache_key]


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


def generate_image(
    *,
    prompt: str,
    output: str = "output.png",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    seed: int = 42,
    model_path: str = "",
    model_key: str = "z-image-turbo",
    low_vram: bool = False,
    reference_style: str = "",
    reference_character: str = "",
) -> ToolResult:
    """Generate a single image using Z-Image model or other supported models.

    Args:
        prompt: Text description of the image to generate.
        output: Output file path for the generated image.
        height: Image height in pixels.
        width: Image width in pixels.
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG guidance scale (0.0 for Turbo).
        seed: Random seed for reproducibility.
        model_path: Path to Z-Image model weights (for z-image-turbo only).
        model_key: Model key for multi-model support (z-image-turbo, flux-schnell, sdxl-turbo, etc.).
        low_vram: Enable low VRAM mode (8-bit quantization + CPU offload).
        reference_style: Style description to prepend to prompt.
        reference_character: Character description to prepend to prompt.

    Returns:
        ToolResult with generated image path and metadata.
    """
    from .core import ReferenceProfile

    try:
        if not prompt.strip():
            return ToolResult(success=False, error="prompt cannot be empty")
        engine = _get_engine(model_path=model_path, low_vram=low_vram, model_key=model_key)
        effective_prompt = prompt
        if reference_style or reference_character:
            ref = ReferenceProfile(
                name="inline",
                style_description=reference_style,
                character_description=reference_character,
            )
            effective_prompt = ref.to_prompt_prefix() + prompt
        img = engine.generate(
            effective_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return ToolResult(
            success=True,
            data={"path": str(out_path)},
            metadata={"version": __version__, "prompt": prompt, "seed": seed},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def batch_generate(
    *,
    prompts_file: str,
    output_dir: str = "outputs",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    start_seed: int = 42,
    model_path: str = "",
    model_key: str = "z-image-turbo",
    low_vram: bool = False,
    reference_style: str = "",
    reference_character: str = "",
    reference_file: str = "",
) -> ToolResult:
    """Batch generate images from a prompts file.

    Args:
        prompts_file: Path to text file with one prompt per line.
        output_dir: Directory to save generated images.
        height: Image height in pixels.
        width: Image width in pixels.
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG guidance scale (0.0 for Turbo).
        start_seed: Starting seed (incremented per image).
        model_path: Path to Z-Image model weights (for z-image-turbo only).
        model_key: Model key for multi-model support (z-image-turbo, flux-schnell, sdxl-turbo, etc.).
        low_vram: Enable low VRAM mode (8-bit quantization + CPU offload).
        reference_style: Style description for all images.
        reference_character: Character description for all images.
        reference_file: Path to JSON reference profile file.

    Returns:
        ToolResult with list of generated image paths.
    """
    from .core import GenerationConfig, ReferenceProfile, load_reference_profile, read_prompts

    try:
        prompts = read_prompts(prompts_file)
        engine = _get_engine(model_path=model_path, low_vram=low_vram, model_key=model_key)
        config = GenerationConfig(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=start_seed,
        )
        ref = None
        if reference_file:
            ref = load_reference_profile(name="batch", from_file=reference_file)
        elif reference_style or reference_character:
            ref = ReferenceProfile(
                name="batch",
                style_description=reference_style,
                character_description=reference_character,
            )
        paths = engine.batch_generate(
            prompts,
            output_dir=output_dir,
            reference=ref,
            config=config,
            start_seed=start_seed,
        )
        return ToolResult(
            success=True,
            data={"paths": paths, "count": len(paths)},
            metadata={"version": __version__, "prompts_file": prompts_file},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def load_reference(
    *,
    name: str,
    style: str = "",
    character: str = "",
    reference_images: Optional[list[str]] = None,
    from_file: str = "",
) -> ToolResult:
    """Load or create a reference profile for consistent character/style.

    Args:
        name: Name for the reference profile.
        style: Style description (e.g. "ink wash painting, traditional Chinese").
        character: Character description (e.g. "a warrior in red armor").
        reference_images: Paths to reference images.
        from_file: Path to JSON file with reference profile.

    Returns:
        ToolResult with reference profile data.
    """
    from .core import load_reference_profile

    try:
        if not name.strip() and not from_file:
            return ToolResult(success=False, error="name or from_file is required")
        ref = load_reference_profile(
            name=name,
            style=style,
            character=character,
            reference_images=reference_images,
            from_file=from_file or None,
        )
        return ToolResult(
            success=True,
            data={
                "name": ref.name,
                "style": ref.style_description,
                "character": ref.character_description,
                "reference_images": ref.reference_image_paths,
            },
            metadata={"version": __version__},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def create_series(
    *,
    name: str,
    theme: str,
    prompts_file: str,
    output_dir: str = "outputs",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    start_seed: int = 42,
    model_path: str = "",
    model_key: str = "z-image-turbo",
    low_vram: bool = False,
    reference_style: str = "",
    reference_character: str = "",
    reference_file: str = "",
) -> ToolResult:
    """Create a themed series of artworks with consistent style/character.

    Args:
        name: Series name.
        theme: Theme description for the series.
        prompts_file: Path to text file with one prompt per line.
        output_dir: Base directory for series output.
        height: Image height in pixels.
        width: Image width in pixels.
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG guidance scale (0.0 for Turbo).
        start_seed: Starting seed.
        model_path: Path to Z-Image model weights (for z-image-turbo only).
        model_key: Model key for multi-model support (z-image-turbo, flux-schnell, sdxl-turbo, etc.).
        low_vram: Enable low VRAM mode (8-bit quantization + CPU offload).
        reference_style: Style description for series consistency.
        reference_character: Character description for series consistency.
        reference_file: Path to JSON reference profile file.

    Returns:
        ToolResult with series manifest and generated paths.
    """
    from .core import (
        GenerationConfig,
        ReferenceProfile,
        SeriesConfig,
        load_reference_profile,
        read_prompts,
    )

    try:
        if not name.strip():
            return ToolResult(success=False, error="name cannot be empty")
        prompts = read_prompts(prompts_file)
        engine = _get_engine(model_path=model_path, low_vram=low_vram, model_key=model_key)
        config = GenerationConfig(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=start_seed,
        )
        ref = None
        if reference_file:
            ref = load_reference_profile(name=name, from_file=reference_file)
        elif reference_style or reference_character:
            ref = ReferenceProfile(
                name=name,
                style_description=reference_style,
                character_description=reference_character,
            )
        series_cfg = SeriesConfig(
            name=name,
            theme=theme,
            prompts=prompts,
            reference=ref,
            config=config,
            output_dir=output_dir,
            start_seed=start_seed,
        )
        result = engine.create_series(series_cfg)
        return ToolResult(
            success=True,
            data=result,
            metadata={"version": __version__},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))
