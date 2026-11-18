---
layout: single
---

# MDH Lowering: From Backend to Backend

The MDH compiler applies final, model-specific optimizations in this stage.
These optimizations are confined to the backend so that target-agnostic and shared code-level decisions remain separate from genuinely target-specific ones.
For example, in CUDA this stage injects CUDA-specific vector data types for memory copy operations and warp-level shuffle operations for combining results across threads.
These transformations have well-defined applicability and are applied deterministically, without empirical tuning.
