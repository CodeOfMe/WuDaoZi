# WuDaoZi

**Batch painting studio powered by Z-Image model** — named after Wu Daozi (吴道子), the greatest painter in Chinese history.

WuDaoZi provides a free and flexible interface to the Z-Image text-to-image model, with special emphasis on **batch artwork generation** under consistent themes, characters, and styles. It supports loading reference profiles (character descriptions, style descriptions, image references) to maintain visual coherence across a series of paintings.

## Features

- **Single image generation** from text prompts using Z-Image model
- **Batch generation** from prompt files with consistent style/character
- **Themed series** creation with manifest JSON tracking all works
- **Reference profiles** for maintaining consistent characters and styles across generations
- **CLI** with full subcommand support (generate, batch, series, reference, gui)
- **PySide6 GUI** for interactive painting studio
- **OpenAI function-calling tools** for agent integration
- **ToolResult API pattern** for programmatic use

## Requirements

- Python >= 3.10
- PyTorch >= 2.5.0
- Z-Image model weights (Turbo recommended)
- CUDA GPU recommended (supports CPU/TPU/MPS)

## Installation

```bash
pip install wudaozi
```

Or install from source:

```bash
cd WuDaoZi
pip install -e .
```

For GUI support:

```bash
pip install wudaozi[gui]
```

For development:

```bash
pip install wudaozi[dev]
```

## Quick Start

### Generate a single image

```bash
wudaozi generate --prompt "A dragon flying over misty mountains in traditional Chinese ink wash style"
```

### Batch generate from prompts file

Create a `prompts.txt` with one prompt per line, then:

```bash
wudaozi batch --prompts prompts.txt --output-dir outputs/ --reference-style "traditional Chinese ink wash painting"
```

### Create a themed series

```bash
wudaozi series --name "Mountain Spirits" --theme "Chinese mythology" --prompts prompts.txt --reference-character "a celestial maiden in flowing white robes"
```

### Launch GUI studio

```bash
wudaozi gui
```

## Usage

### CLI

```
wudaozi [-V] [-v] [-q] [--json] [-o PATH] <command>

Commands:
  generate    Generate a single image
  batch       Batch generate from prompts file
  series      Create a themed artwork series
  reference   Create a reference profile JSON
  gui         Launch GUI studio

Flags:
  -V, --version   Show version
  -v, --verbose   Verbose output
  -q, --quiet     Suppress non-essential output
  --json          Output as JSON
  -o, --output    Output path
```

### Generate

```bash
wudaozi generate \
  --prompt "A phoenix rising from flames" \
  --height 1024 --width 1024 \
  --steps 8 --seed 42 \
  --reference-style "traditional Chinese painting" \
  --reference-character "golden phoenix with fiery tail feathers"
```

### Batch

```bash
wudaozi batch \
  --prompts my_prompts.txt \
  --output-dir outputs/ \
  --reference-style "ink wash painting, Song dynasty style" \
  --start-seed 100
```

### Series

```bash
wudaozi series \
  --name "Four Seasons" \
  --theme "Landscape through the seasons" \
  --prompts four_seasons.txt \
  --reference-file hero_profile.json
```

### Reference

```bash
wudaozi reference \
  --name "Hero" \
  --style "ink wash painting, expressive brushstrokes" \
  --character "a wandering swordsman with a bamboo hat and flowing robes"
```

## Python API

```python
from wudaozi import generate_image, batch_generate, load_reference, create_series

# Generate a single image
result = generate_image(
    prompt="A dragon flying over mountains",
    output="dragon.png",
    reference_style="ink wash painting",
)
print(result.success)    # True
print(result.data)       # {"path": "dragon.png"}
print(result.metadata)   # {"version": "1.0.0", ...}

# Batch generate
result = batch_generate(
    prompts_file="prompts.txt",
    output_dir="outputs/",
    reference_character="a warrior in red armor",
)
print(result.data["count"])  # number of images

# Create a reference profile
result = load_reference(
    name="Hero",
    style="ink wash painting",
    character="a warrior in red armor",
)

# Create a themed series
result = create_series(
    name="Mountain Spirits",
    theme="Chinese mythology",
    prompts_file="prompts.txt",
    reference_style="traditional landscape painting",
)
```

## Agent Integration (OpenAI Function Calling)

```python
from openai import OpenAI
from wudaozi.tools import TOOLS, dispatch

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Generate a mountain landscape in Chinese ink style"}],
    tools=TOOLS,
)

message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        result = dispatch(tool_call.function.name, tool_call.function.arguments)
        print(result)
```

## Reference Profiles

Create a JSON file for consistent character/style across generations:

```json
{
  "name": "Wandering Hero",
  "style": "ink wash painting, Song dynasty landscape style, expressive brushstrokes",
  "character": "a lone swordsman in flowing white robes, bamboo hat, carrying a long blade",
  "reference_images": ["hero_ref_1.png", "hero_ref_2.png"],
  "metadata": {
    "created_by": "WuDaoZi",
    "notes": "Main character for the 'Jianghu Legends' series"
  }
}
```

## Development

```bash
pip install wudaozi[dev]
ruff format . && ruff check . && mypy . && pytest tests/test_unified_api.py -v
```

## PyPI Publishing

Bump version (defined in `wudaozi/__init__.py`), build, and upload:

```bash
./upload_pypi.sh
```

On Windows:

```batch
upload_pypi.bat
```

## License

GPLv3