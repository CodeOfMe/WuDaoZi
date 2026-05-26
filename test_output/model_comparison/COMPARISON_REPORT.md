# WuDaoZi 模型对比测试报告

## 测试环境

| 项目 | 规格 |
|------|------|
| **GPU** | AMD Radeon RX 7900 XTX 24GB |
| **ROCm** | 7.2 (HIP 7.2.26015) |
| **PyTorch** | 2.11.0+rocm7.2 |
| **Diffusers** | 0.38.0 |
| **Python** | 3.12.13 |
| **测试日期** | 2026-05-17 |

## 测试条件

- **提示词**: "A beautiful traditional Chinese landscape painting with mountains, rivers, and a small boat"
- **随机种子**: 42
- **图像尺寸**: 1024x1024
- **所有模型使用相同的提示词和种子**

## 模型对比结果

### SDXL-Turbo

| 指标 | 数值 |
|------|------|
| **Pipeline** | StableDiffusionXLPipeline |
| **模型来源** | ModelScope (stabilityai/sdxl-turbo) |
| **模型大小** | 39 GB |
| **加载时间** | 5.32 秒 |
| **生成步数** | 4 步 |
| **生成时间** | 2.02 秒 |
| **峰值显存** | 9.76 GB |
| **Guidance Scale** | 0.0 (Turbo 模式) |
| **输出图像** | `sdxl-turbo.png` (1.7MB) |

### Kolors

| 指标 | 数值 |
|------|------|
| **Pipeline** | KolorsPipeline |
| **模型来源** | ModelScope (Kwai-Kolors/Kolors) |
| **模型大小** | 27 GB |
| **加载时间** | 9.55 秒 |
| **生成步数** | 20 步 |
| **生成时间** | 6.54 秒 |
| **峰值显存** | 19.91 GB |
| **Guidance Scale** | 7.5 |
| **输出图像** | `kolors.png` (1.7MB) |

## 性能对比

| 模型 | 加载时间 | 生成时间 | 峰值显存 | 速度比 | 显存比 |
|------|----------|----------|----------|--------|--------|
| **SDXL-Turbo** | 5.32s | 2.02s | 9.76GB | 1x (基准) | 1x (基准) |
| **Kolors** | 9.55s | 6.54s | 19.91GB | 3.2x 慢 | 2.0x 高 |

## 对比图

对比网格图已保存至: `test_output/model_comparison/comparison_grid.png`

## 结论

1. **SDXL-Turbo** 是速度最快的选择，仅需 2 秒即可生成 1024x1024 图像，显存占用仅 9.76GB
2. **Kolors** 生成质量可能更好（20 步 vs 4 步），但速度慢 3.2 倍，显存占用高 2 倍
3. 两个模型都在 ROCm 7900 XTX 上正常工作
4. Z-Image-Turbo 和 PixArt-Alpha 因 ModelScope 下载问题暂不可用

## 使用命令

```bash
conda activate rocm

# SDXL-Turbo (快速)
wudaozi generate --prompt "水墨山水画" --model-key sdxl-turbo

# Kolors (高质量)
wudaozi generate --prompt "水墨山水画" --model-key kolors
```
