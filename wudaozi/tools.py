"""OpenAI function-calling tools for WuDaoZi."""

import json
from typing import Any

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "wudaozi_generate_image",
            "description": (
                "Generate a single image from a text prompt using Z-Image model. "
                "Supports style and character references for consistent artwork."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate",
                    },
                    "output": {
                        "type": "string",
                        "description": "Output file path for the generated image",
                        "default": "output.png",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels",
                        "default": 1024,
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels",
                        "default": 1024,
                    },
                    "num_inference_steps": {
                        "type": "integer",
                        "description": "Number of denoising steps (8 for Turbo model)",
                        "default": 8,
                    },
                    "guidance_scale": {
                        "type": "number",
                        "description": "CFG guidance scale, use 0.0 for Turbo model",
                        "default": 0.0,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility",
                        "default": 42,
                    },
                    "model_path": {
                        "type": "string",
                        "description": "Path to Z-Image model weights (env: WUDAOZI_MODEL_PATH, auto-detect if empty)",
                        "default": "",
                    },
                    "reference_style": {
                        "type": "string",
                        "description": (
                            "Style description to prepend to prompt (e.g. 'ink wash painting, traditional Chinese')"
                        ),
                        "default": "",
                    },
                    "reference_character": {
                        "type": "string",
                        "description": "Character description to prepend to prompt (e.g. 'a warrior in red armor')",
                        "default": "",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wudaozi_batch_generate",
            "description": (
                "Batch generate images from a prompts text file. "
                "All images share consistent style and character if references are provided."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompts_file": {
                        "type": "string",
                        "description": "Path to text file with one prompt per line",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save generated images",
                        "default": "outputs",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels",
                        "default": 1024,
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels",
                        "default": 1024,
                    },
                    "num_inference_steps": {
                        "type": "integer",
                        "description": "Number of denoising steps",
                        "default": 8,
                    },
                    "guidance_scale": {
                        "type": "number",
                        "description": "CFG guidance scale",
                        "default": 0.0,
                    },
                    "start_seed": {
                        "type": "integer",
                        "description": "Starting seed, incremented per image",
                        "default": 42,
                    },
                    "model_path": {
                        "type": "string",
                        "description": "Path to Z-Image model weights (env: WUDAOZI_MODEL_PATH, auto-detect if empty)",
                        "default": "",
                    },
                    "reference_style": {
                        "type": "string",
                        "description": "Style description for all images in the batch",
                        "default": "",
                    },
                    "reference_character": {
                        "type": "string",
                        "description": "Character description for all images in the batch",
                        "default": "",
                    },
                    "reference_file": {
                        "type": "string",
                        "description": "Path to JSON reference profile file",
                        "default": "",
                    },
                },
                "required": ["prompts_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wudaozi_load_reference",
            "description": (
                "Create or load a reference profile for maintaining "
                "consistent style and character across multiple generations"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the reference profile",
                    },
                    "style": {
                        "type": "string",
                        "description": "Style description (e.g. 'ink wash painting, traditional Chinese')",
                        "default": "",
                    },
                    "character": {
                        "type": "string",
                        "description": "Character description (e.g. 'a warrior in red armor')",
                        "default": "",
                    },
                    "reference_images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to reference images",
                        "default": [],
                    },
                    "from_file": {
                        "type": "string",
                        "description": "Path to JSON file with reference profile",
                        "default": "",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wudaozi_create_series",
            "description": (
                "Create a themed series of artworks with consistent style and character. "
                "Generates a series manifest JSON alongside all images."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Series name",
                    },
                    "theme": {
                        "type": "string",
                        "description": "Theme description for the series",
                    },
                    "prompts_file": {
                        "type": "string",
                        "description": "Path to text file with one prompt per line",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Base directory for series output",
                        "default": "outputs",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels",
                        "default": 1024,
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels",
                        "default": 1024,
                    },
                    "num_inference_steps": {
                        "type": "integer",
                        "description": "Number of denoising steps",
                        "default": 8,
                    },
                    "guidance_scale": {
                        "type": "number",
                        "description": "CFG guidance scale",
                        "default": 0.0,
                    },
                    "start_seed": {
                        "type": "integer",
                        "description": "Starting seed",
                        "default": 42,
                    },
                    "model_path": {
                        "type": "string",
                        "description": "Path to Z-Image model weights (env: WUDAOZI_MODEL_PATH, auto-detect if empty)",
                        "default": "",
                    },
                    "reference_style": {
                        "type": "string",
                        "description": "Style description for series consistency",
                        "default": "",
                    },
                    "reference_character": {
                        "type": "string",
                        "description": "Character description for series consistency",
                        "default": "",
                    },
                    "reference_file": {
                        "type": "string",
                        "description": "Path to JSON reference profile file",
                        "default": "",
                    },
                },
                "required": ["name", "theme", "prompts_file"],
            },
        },
    },
]


def dispatch(name: str, arguments: dict[str, Any] | str) -> dict:
    """Dispatch tool call to appropriate API function.

    Args:
        name: Tool name from LLM response.
        arguments: Tool arguments (dict or JSON string).

    Returns:
        Dict representation of ToolResult.

    Raises:
        ValueError: Unknown tool name.
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if name == "wudaozi_generate_image":
        from .api import generate_image

        result = generate_image(**arguments)
        return result.to_dict()

    if name == "wudaozi_batch_generate":
        from .api import batch_generate

        result = batch_generate(**arguments)
        return result.to_dict()

    if name == "wudaozi_load_reference":
        from .api import load_reference

        result = load_reference(**arguments)
        return result.to_dict()

    if name == "wudaozi_create_series":
        from .api import create_series

        result = create_series(**arguments)
        return result.to_dict()

    raise ValueError(f"Unknown tool: {name}")
