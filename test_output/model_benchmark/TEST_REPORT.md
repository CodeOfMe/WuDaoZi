# WuDaoZi 模型测试总结报告

## 测试环境

| 项目 | 规格 |
|------|------|
| **GPU** | AMD Radeon RX 7900 XTX 24GB |
| **ROCm** | 7.2 (HIP 7.2.26015) |
| **PyTorch** | 2.11.0+rocm7.2 |
| **Diffusers** | 0.38.0 |
| **Transformers** | 4.57.6 |
| **ModelScope** | 1.36.3 |
| **Python** | 3.12.13 |
| **Conda 环境** | rocm |
| **测试日期** | 2026-05-14 |

## 系统信息

- **GPU 显存**: 23.98 GB
- **GPU 计算单元**: 48 个多处理器
- **Compute Capability**: 11.0
- **计算精度**: torch.float16 (ROCm 自动选择)
- **ROCm 优化**: `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 自动启用

## 模型下载与测试结果

### 1. SDXL-Turbo ✅ 完全成功

| 项目 | 详情 |
|------|------|
| **ModelScope ID** | `stabilityai/sdxl-turbo` |
| **模型大小** | 39 GB (完整) |
| **Pipeline 类** | StableDiffusionXLPipeline |
| **加载时间** | 4.1 秒 |
| **生成时间** | 3.5 秒 (8 steps, 1024x1024) |
| **峰值显存** | 9.8 GB |
| **输出图像** | `test_output/model_benchmark/sdxl-turbo_test.png` (1.7MB, 1024x1024) |
| **状态** | ✅ **完全正常工作** |

**测试命令**:
```bash
wudaozi generate --prompt "A beautiful Chinese landscape" --model-key sdxl-turbo
```

### 2. Z-Image-Turbo ❌ 下载不完整

| 项目 | 详情 |
|------|------|
| **ModelScope ID** | `Tongyi-MAI/Z-Image-Turbo` |
| **模型大小** | 5.2 GB (不完整) |
| **Pipeline 类** | ZImagePipeline |
| **问题** | transformer 分片权重文件缺失 |
| **状态** | ❌ 无法加载 |

**原因**: ModelScope 上的 Z-Image-Turbo 模型缺少 `diffusion_pytorch_model-00001-of-00003.safetensors` 等分片文件。原始 Tongyi-MAI 仓库也有同样问题。

**解决方案**: 需要从 HuggingFace 完整下载，或联系模型作者修复 ModelScope 仓库。

### 3. PixArt-Alpha ❌ 格式错误

| 项目 | 详情 |
|------|------|
| **ModelScope ID** | `AI-ModelScope/PixArt-alpha` |
| **模型大小** | 37 GB (ComfyUI 格式) |
| **问题** | 下载的是 ComfyUI .pth 格式，不是 diffusers 格式 |
| **Diffusers 下载** | 1.8 GB (不完整，进行中) |
| **状态** | ❌ 无法加载 |

**原因**: ModelScope 上的 PixArt-Alpha 是 ComfyUI 格式，包含 `.pth` 权重文件和 ComfyUI 工作流 JSON。diffusers 格式的下载正在进行中。

**解决方案**: 使用 HuggingFace 的 `PixArt-alpha/PixArt-XL-2-1024-MS` 仓库下载 diffusers 格式。

### 4. Kolors ❌ 依赖缺失

| 项目 | 详情 |
|------|------|
| **ModelScope ID** | `Kwai-Kolors/Kolors` |
| **模型大小** | 27 GB (完整) |
| **Pipeline 类** | KolorsPipeline |
| **问题** | 缺少 sentencepiece 库 |
| **状态** | ❌ 测试时失败 (已安装 sentencepiece) |

**原因**: KolorsPipeline 需要 `sentencepiece` 库，测试时未安装。

**解决方案**: 已安装 `sentencepiece`，下次测试应该可以正常工作。

## 代码改进总结

### 已完成的改进

1. **统一 models/ 目录**: 所有模型下载到 `<package>/models/` 目录
2. **ROCm 自动检测**: 自动识别 AMD GPU 并选择 float16 精度
3. **ModelScope 主下载源**: 修正了模型 ID，不再回退到 HuggingFace
4. **下载优化**: 跳过 ONNX 等大文件，只下载推理必需文件
5. **统一接口**: MultiModelEngine 支持所有模型类型
6. **内存优化**: ROCm 环境下自动启用 expandable_segments

### 待解决的问题

1. **Z-Image-Turbo**: 需要从 HuggingFace 完整下载分片权重
2. **PixArt-Alpha**: 需要完成 diffusers 格式下载
3. **Kolors**: 需要重新测试（sentencepiece 已安装）

## 使用指南

### 激活环境
```bash
conda activate rocm
```

### 下载模型
```bash
# 从 ModelScope 下载
wudaozi models --download sdxl-turbo

# 查看已下载模型
wudaozi models --downloaded
```

### 生成图片
```bash
# 使用 SDXL-Turbo (已验证可用)
wudaozi generate --prompt "水墨山水画" --model-key sdxl-turbo

# 批量生成
wudaozi batch --prompts prompts.txt --model-key sdxl-turbo --output-dir outputs/
```

### Python API
```python
from wudaozi.model_manager import MultiModelEngine

engine = MultiModelEngine(model_key='sdxl-turbo')
engine.load()
img = engine.generate("A beautiful landscape", height=1024, width=1024, seed=42)
img.save("output.png")
```

## 性能基准 (SDXL-Turbo on 7900 XTX)

| 指标 | 数值 |
|------|------|
| 加载时间 | 4.1 秒 |
| 生成时间 (4 steps) | ~2 秒 |
| 生成时间 (8 steps) | 3.5 秒 |
| 峰值显存 | 9.8 GB |
| 图像尺寸 | 1024x1024 |
| 输出格式 | PNG |

## 文件结构

```
WuDaoZi/
├── models/
│   ├── z-image-turbo/        # 5.2GB (不完整)
│   ├── sdxl-turbo/           # 39GB (完整) ✅
│   ├── kolors/               # 27GB (完整)
│   ├── pixart-alpha/         # 37GB (ComfyUI 格式)
│   └── pixart-alpha-diffusers/ # 1.8GB (下载中)
├── test_output/
│   └── model_benchmark/
│       ├── sdxl-turbo_test.png  # 测试生成的图片
│       └── benchmark_report.json # 完整测试报告
└── wudaozi/
    ├── model_manager.py      # 模型管理 (已改进)
    ├── core.py               # 核心引擎 (已改进)
    └── ...
```

## 结论

- **SDXL-Turbo** 在 ROCm 7900 XTX 上完全正常工作，性能优秀
- 其他模型需要修复下载问题后才能使用
- 代码框架已支持多模型统一接口和 ROCm 优化
- 建议优先修复 Z-Image-Turbo 的下载问题（默认模型）
