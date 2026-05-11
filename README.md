# scmp_kernels

Shared stochastic-computing kernels and mixed-precision dispatch, factored out of `scmp_llm` and `vit_sc`.

## Layout

```
scmp_kernels/
├── sc/             # Stochastic-computing kernels (Triton)        ← migrated
├── mp/             # Mixed-precision dispatch + config             (placeholder)
├── qwt/            # QwT compensation                              (placeholder)
└── sensitivity/    # Per-(op, block) sensitivity tools             (placeholder)
```

## SC quickstart

```python
import torch
from scmp_kernels.sc import sc_matmul

a = torch.randn(128, 1024, device="cuda")
b = torch.randn(512, 1024, device="cuda")

# Per-row quantization (most common — used by all linear/MLP paths)
y = sc_matmul(a, b, granularity="per_row", sc_prec=8)

# Per-tensor quantization
y = sc_matmul(a, b, granularity="per_tensor", sc_prec=8)

# Per-head batched (QK attention pattern)
q = torch.randn(16, 196, 64, device="cuda")   # (BH, N, D)
k = torch.randn(16, 196, 64, device="cuda")
y = sc_matmul(q, k, granularity="per_head", sc_prec=8)

# MLP fast path: per-row + chunk_d on wide D
y = sc_matmul(a, b, granularity="per_row", chunk_d=72, sc_prec=8)
```

### API

```
sc_matmul(a, b,
    granularity: "per_tensor" | "per_row" | "per_head" = "per_row",
    mode: "bipolar" | "unipolar" = "bipolar",
    sc_prec: int = 8,
    stoc_len: int | None = None,    # default 2 ** sc_prec
    chunk_d: int = 0,               # only valid for granularity="per_row" + mode="bipolar"
    config: dict | None = None,     # Sobol config; auto-built if None
) -> torch.Tensor
```

`chunk_d > 0` requires `granularity="per_row"` and `mode="bipolar"`. Other combinations raise `ValueError`.

## MP / QwT / Sensitivity

Not yet migrated. The empty `mp/`, `qwt/`, `sensitivity/` packages reserve the namespace.
