# hypbr relief patch for Depth Anything V2 Small

`da2s_relief_pbr.safetensors` holds the DPT neck and head of
[Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) (2,728,513 parameters, fp16),
fine-tuned by hypbr to predict surface relief from close-up material photos. The DINOv2 ViT-S backbone is unchanged and
is downloaded from Hugging Face on first use; `da2s_config.json` is that model's configuration.

- **Base model**: Depth Anything V2 Small, Copyright the Depth Anything authors, licensed under the
  [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). These tensors are modified versions of its neck
  and head weights. (Only the Small model is Apache-2.0; the Base, Large and Giant models are CC BY-NC 4.0.)
- **Training data**: 344 CC0 scans from [Poly Haven](https://polyhaven.com) (height + normal + albedo), rendered as
  studio-lit photos with randomized raking light, plus flat painted patterns that must stay flat.
- **Recipe**: frozen backbone; multi-scale height-correlation, normal-slope and high-frequency losses; EMA weights and
  an L2 pull toward the original weights. 10 minutes on an Apple M1.

Held-out results (20 Poly Haven scans never trained on, 7 ambientCG scans from a different lab, 2 flat traps; score =
0.35 depth + 0.35 slope + 0.30 fine-detail correlation, minus a penalty for reading paint as relief):

| Model | Score | Cross-lab depth / slope / detail | Paint read as relief |
|---|---|---|---|
| Apple Depth Pro 2-S, unmodified | 0.027 | -0.015 / 0.021 / 0.014 | 0.07 |
| Apple Depth Pro 2-S, best hypbr fine-tune | 0.233 | 0.269 / 0.271 / 0.320 | 0.28 |
| Depth Anything V2 Small, unmodified | 0.183 | 0.255 / 0.271 / 0.328 | 0.02 |
| **Depth Anything V2 Small + this patch** | **0.266** | **0.350 / 0.295 / 0.355** | 0.07 |
