# hypbr Neural Patches (Apache-2.0 & CC0 Derived)

1. **`da2s_relief_pbr.safetensors`** (5.21 MB, 2,728,513 params, fp16): DPT neck + head of
   [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) (Apache-2.0),
   fine-tuned with broad-shape distillation (`exp030_distill_1800s_s202`, 13,639 steps on 344 CC0 Poly Haven scans).
   Single-head inference (`BLEND_LOW = 0.0`) outperforms the earlier two-head runtime blend while halving decoder work.

2. **`svbrdf_da2s.safetensors`** (3.15 MB, 1,649,669 params, fp16): optional skip-connected SVBRDF decoder
   (`hypbr/svbrdf_da2.py`) sharing the frozen DINOv2 ViT-S backbone to predict shadow-free delighted albedo,
   calibrated roughness, and metalness, trained on 882 CC0 Poly Haven + ambientCG SVBRDF scans.

3. **`microunet_dh_1k_f16.bin`** (229.6 KB, 117,561 params, fp16): optional 3-scale curl-free micro-height U-Net
   (`hypbr/micro_detail.py` & WebGPU compute) trained on 1K CC0 scans with VGG16 perceptual + PatchGAN losses.

Held-out relief results (20 Poly Haven scans never trained on, 7 ambientCG scans from a different lab, 2 flat traps):

| Model | Res | Val Score | Cross-lab depth / slope / detail | Paint leak |
|---|---:|---:|---|---:|
| Apple Depth Pro 2-S, unmodified | 512 | 0.027 | -0.015 / 0.021 / 0.014 | 0.070 |
| Apple Depth Pro 2-S, best hypbr fine-tune | 512 | 0.233 | 0.269 / 0.271 / 0.320 | 0.280 |
| Depth Anything V2 Small, unmodified | 518 | 0.183 | 0.255 / 0.271 / 0.328 | 0.020 |
| Earlier `da2s` patch + 0.3 two-head blend | 518 | 0.283 | 0.351 / 0.290 / 0.353 | 0.070 |
| **Shipped `da2s_relief_pbr` (single head)** | **518** | **0.328** | **0.360 / 0.319 / 0.374** | **0.029** |
| **Shipped `da2s_relief_pbr` (single head)** | **1036** | **0.305** | **0.367 / 0.330 / 0.365** | **0.033** |
| Optional `da3mono` Pro (`exp032`, Apache-2.0) | 1008 | 0.325 | **0.405 / 0.365 / 0.428** | **0.010** |
