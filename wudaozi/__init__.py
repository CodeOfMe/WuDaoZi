"""WuDaoZi - Batch painting studio powered by Z-Image model."""

__version__ = "1.0.2"

__all__ = ["__version__"]


def __getattr__(name: str):
    if name == "ToolResult":
        from .api import ToolResult

        return ToolResult
    if name == "generate_image":
        from .api import generate_image

        return generate_image
    if name == "batch_generate":
        from .api import batch_generate

        return batch_generate
    if name == "load_reference":
        from .api import load_reference

        return load_reference
    if name == "create_series":
        from .api import create_series

        return create_series
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
