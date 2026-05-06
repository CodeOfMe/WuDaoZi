# WuDaoZi 吴道子

**基于 Z-Image 模型的批量绘画工作室** —— 以中国历史上最伟大的画家吴道子命名。

WuDaoZi 提供了一个自由灵活的接口来使用 Z-Image 文生图模型，特别擅长**在保持主题、角色和风格一致的前提下批量绘制作品**。支持加载参考画像（角色描述、风格描述、参考图像），以在系列作品中保持视觉连贯性。

## 特性

- **单图生成**：使用 Z-Image 模型从文字描述生成图像
- **批量生成**：从提示词文件批量生成，保持一致的风格/角色
- **主题系列**：创建主题系列作品，自动生成清单 JSON 文件
- **参考画像**：加载角色描述、风格描述，在多幅作品中保持一致
- **CLI 命令行**：完整子命令支持（generate, batch, series, reference, gui）
- **PySide6 GUI**：交互式绘画工作室
- **OpenAI 函数调用工具**：支持智能体集成
- **ToolResult API 模式**：方便编程使用

## 系统要求

- Python >= 3.10
- PyTorch >= 2.5.0
- Z-Image 模型权重（推荐 Turbo 版本）
- 推荐 CUDA GPU（也支持 CPU/TPU/MPS）

## 安装

```bash
pip install wudaozi
```

或从源码安装：

```bash
cd WuDaoZi
pip install -e .
```

GUI 支持：

```bash
pip install wudaozi[gui]
```

开发环境：

```bash
pip install wudaozi[dev]
```

## 快速开始

### 生成单张图像

```bash
wudaozi generate --prompt "一条龙飞过云雾缭绕的山峰，传统水墨画风格"
```

### 从提示词文件批量生成

创建 `prompts.txt`，每行一个提示词：

```bash
wudaozi batch --prompts prompts.txt --output-dir outputs/ --reference-style "传统中国水墨画"
```

### 创建主题系列

```bash
wudaozi series --name "山灵" --theme "中国神话" --prompts prompts.txt --reference-character "穿白色飘袍的仙女"
```

### 启动 GUI 工作室

```bash
wudaozi gui
```

## 使用方法

### 命令行

```
wudaozi [-V] [-v] [-q] [--json] [-o PATH] <命令>

命令：
  generate    生成单张图像
  batch       从提示词文件批量生成
  series      创建主题系列作品
  reference   创建参考画像 JSON
  gui         启动 GUI 工作室

标志：
  -V, --version   显示版本
  -v, --verbose   详细输出
  -q, --quiet     静默模式
  --json          JSON 格式输出
  -o, --output    输出路径
```

### 生成

```bash
wudaozi generate \
  --prompt "凤凰涅槃" \
  --height 1024 --width 1024 \
  --steps 8 --seed 42 \
  --reference-style "传统中国画" \
  --reference-character "金色凤凰，火焰尾羽"
```

### 批量生成

```bash
wudaozi batch \
  --prompts my_prompts.txt \
  --output-dir outputs/ \
  --reference-style "水墨画，宋代风格" \
  --start-seed 100
```

### 系列创作

```bash
wudaozi series \
  --name "四季山水" \
  --theme "四季流转的山水" \
  --prompts four_seasons.txt \
  --reference-file hero_profile.json
```

### 参考画像

```bash
wudaozi reference \
  --name "侠客" \
  --style "水墨画，写意笔法" \
  --character "戴着斗笠、身穿飘飘白衣的独行剑客"
```

## Python API

```python
from wudaozi import generate_image, batch_generate, load_reference, create_series

# 生成单张图像
result = generate_image(
    prompt="龙飞越过山峦",
    output="dragon.png",
    reference_style="水墨画",
)
print(result.success)    # True
print(result.data)       # {"path": "dragon.png"}

# 批量生成
result = batch_generate(
    prompts_file="prompts.txt",
    output_dir="outputs/",
    reference_character="红甲战士",
)
print(result.data["count"])  # 生成图像数量

# 创建参考画像
result = load_reference(
    name="侠客",
    style="水墨画",
    character="红甲战士",
)

# 创建主题系列
result = create_series(
    name="山灵",
    theme="中国神话",
    prompts_file="prompts.txt",
    reference_style="传统山水画",
)
```

## 智能体集成（OpenAI 函数调用）

```python
from openai import OpenAI
from wudaozi.tools import TOOLS, dispatch

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "生成一幅水墨风格的山水画"}],
    tools=TOOLS,
)

message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        result = dispatch(tool_call.function.name, tool_call.function.arguments)
        print(result)
```

## 参考画像文件

创建 JSON 文件以在多次生成中保持角色/风格一致：

```json
{
  "name": "游侠",
  "style": "水墨画，宋代山水风格，写意笔法",
  "character": "戴着斗笠、身穿飘白袍、佩长刀的独行剑客",
  "reference_images": ["hero_ref_1.png", "hero_ref_2.png"],
  "metadata": {
    "created_by": "WuDaoZi",
    "notes": "《江湖传奇》系列主角"
  }
}
```

## 开发

```bash
pip install wudaozi[dev]
ruff format . && ruff check . && mypy . && pytest tests/test_unified_api.py -v
```

## PyPI 发布

自动递增版本号（定义在 `wudaozi/__init__.py`）、构建并上传：

```bash
./upload_pypi.sh
```

Windows：

```batch
upload_pypi.bat
```

## 许可证

GPLv3