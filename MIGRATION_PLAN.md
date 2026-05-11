# scmp_kernels Migration Plan

**Date**: 2026-05-10
**Sources audited**: `scmp_llm/SC/` and `vit_sc/sc/` (+ `vit_sc/sc_integration/`)
**Destination**: `scmp_kernels/` (currently empty)

This document is the keep/merge/drop decision for moving the shared stochastic-computing kernels into `scmp_kernels`. It is **not** an experiment plan — it is a code-consolidation audit. Both downstream repos will later depend on `scmp_kernels` instead of carrying their own `SC/` or `sc/` copies.

---

## TL;DR

**The library splits cleanly into two sub-packages.** Audit of every internal import shows SC kernels and MP dispatch never import each other inside the library — they only meet at the user layer (e.g. `sc_integration/sc_attention.py`). Recommended layout:

```
scmp_kernels/src/scmp_kernels/
├── sc/      # stochastic computing kernels — no MP awareness
│   sc_triton.py, config_helpers.py, sng.py, rng.py, lfsr_taps.py, constants.py
├── mp/      # mixed-precision dispatch — no SC awareness
│   config.py, auto_calibrator.py
└── __init__.py   # re-export public API from both
```

**Per-file provenance:**

| Target | Source | Provenance |
|---|---|---|
| `sc/sc_triton.py` | scmp_llm + vit_sc | MERGE (scmp_llm base + vit_sc's batched grouped path + det_kernel_tuning) |
| `sc/config_helpers.py` | scmp_llm + vit_sc | MERGE (vit_sc superset has 2 extra Sobol variants) |
| `sc/sng.py` | scmp_llm | TAKE scmp_llm (marginally newer; functionally equivalent) |
| `sc/rng.py` | either | IDENTICAL |
| `sc/lfsr_taps.py` | either | IDENTICAL |
| `sc/constants.py` | either | extract `FP8_E4M3_MAX`/`FP8_E5M2_MAX`/`INT8_MAX` from current `sc.py` |
| `mp/config.py` | scmp_llm + vit_sc | TAKE scmp_llm base + port vit_sc's `fixed_levels`, `set_current_block_idx`, `_classify_all_rows_to_level`; keep `FreeBoundaryMPConfig` as thin subclass (see [mp section](#mpconfigpy)) |
| `mp/auto_calibrator.py` | vit_sc | TAKE vit_sc (scmp_llm has no equivalent) |

**Drop entirely** — confirmed zero callers:

- `sc.py` (reference numpy `matmul_sc`, 0 refs in either repo)
- `dse.py` (0 external refs in either repo)
- `sc_enable.py` (scmp_llm only; superseded by Triton enable path)
- `noise_model_calibration.py` (scmp_llm only; 0 refs)
- `SC/bench_*.py`, `SC/compare_*.py`, `SC/test_kernel_opt.py` — these are bench/validation scripts; if useful, relocate to `scmp_llm/experiments/` (not into `scmp_kernels`)

**Public API surface that `scmp_kernels` must export** (this is what downstream code imports today):

```python
from scmp_kernels import (
    # core matmul entrypoints
    sc_matmul,
    sc_matmul_mlp,
    sc_matmul_grouped,
    sc_matmul_enable_triton,
    sc_matmul_enable_triton_mlp,
    sc_matmul_grouped_enable_triton,
    sc_matmul_grouped_enable_triton_batched,   # vit_sc-only today; safe to expose
    sc_matmul_enable_batched_bipolar,
    # config + dispatch
    make_sobol_simple_config,
    classify_rows_by_metric,
    adaptive_classify_rows,
    # calibration (vit_sc-only today)
    RidgeFitter,
    AutoMPBudgetLogger,
    auto_calibrate_mp,
)
```

Everything else (RNGPool, SNGBank, Sobol, LFSR, lfsr tap tables, all `@triton.jit` kernels, all `make_*_config` variants beyond `make_sobol_simple_config`) is internal — should not be in the public namespace.

---

## Per-file detail

### sc_triton.py

Both copies are ~150 KB / ~100+ top-level symbols. Public matmul entrypoints (what `sc_integration/` and Q-DiT actually call) are listed above. Everything else is internal `@triton.jit` kernels and helpers — keep all of those that are reachable from the public entrypoints; delete the dead ones below.

**Dead in both repos** (no caller anywhere outside `sc_triton.py` itself):

- `bin_to_stoc_packed`, `bin_to_stoc_packed_unipolar`
- `xnor_matmul`, `and_matmul`, `fused_xnor_matmul`, `fused_and_matmul`
- `build_enable_tables`, `build_k_table_only`, `enable_matmul_triton`, `enable_matmul_compact`, `enable_matmul_compact_mlp`
- `matmul_sc_triton`, `matmul_sc_triton_from_saved`
- `clear_rng_cache`
- `test_all_configs`, `test_sc_matmul`, `benchmark_comparison`

**Conflict to resolve**: `fused_quantize_bipolar` / `fused_quantize_unipolar` are reported DEAD by the vit_sc-side survey but USED by `scmp_llm/calibrate_mp_thresholds.py`. **Decision**: keep them — calibration in scmp_llm still imports them.

**Merge needed** (scmp_llm's copy is 1.5 days newer, but each side has unique kernels):

| In scmp_llm only | In vit_sc only |
|---|---|
| `_bit_reverse`, `_owen_scramble`, `_prepare_rng_prefix`, `_resolve_rng_levels` (fresh RNG helpers) | `det_kernel_tuning` class + `_det_kernel_tuning_active`, `_grouped_symmetric_quant_batched`, block-sizing helpers |
| | `sc_matmul_grouped_enable_triton_batched` (vit_sc batched path) |

**Recommended base**: scmp_llm `sc_triton.py` (newer mtime, fresher RNG infra). Port the four vit_sc-only items on top (1 public function + 1 tuning class + 1 quant kernel + small helpers). Both sides' new code is additive — no semantic conflicts expected.

### config_helpers.py

| In scmp_llm | In vit_sc |
|---|---|
| 13 functions, 698 LOC | 15 functions, 763 LOC (adds `make_sobol_antithetic_config`, `make_sobol_altseed_config`) |

**Merge needed**: take vit_sc's superset (it strictly adds two Sobol config variants); only the `make_sobol_simple_config` path is currently CORE, but keeping the other two costs nothing and is consistent with the auto-calibrator code path.

**Internal dead** (no external caller in either repo): all `make_*_config` variants except `make_sobol_simple_config`, `make_sobol_dse_config`, and the two vit_sc-only Sobol ones. Decision: drop the dead variants in `scmp_kernels` — they're easy to resurrect from git if needed.

### mp/config.py (formerly `mp_config.py`)

**scmp_llm is the more general representation; vit_sc is a special case of it.**

scmp_llm's `AdaptiveMPConfig` has parametric granularity over `(operator × timestep × layer)`:

```python
bucket_thresholds: dict[tuple[str, int, int], list[float]]   # (op, t_bucket, l_bucket) → thresholds
timestep_buckets: int = 1     # 1 = coarsest along t; T = finest
layer_buckets:    int = 1     # 1 = coarsest along l; L = finest
```

vit_sc's `FreeBoundaryMPConfig` stores `boundaries[(block_idx, op_name)] → tensor` with no timestep axis and no coarsening on the block axis.

The math degenerates: with `_bucket_index(block_idx, total_blocks, layer_buckets=total_blocks)` returning `block_idx` itself, `bucket_thresholds[(op, 0, block_idx)]` ≡ vit_sc's `boundaries[(block_idx, op)]`. Same threshold-list semantics on a `[0,1]`-normalized metric (`_classify_rows_by_thresholds` ≡ `_classify_with_free_boundaries`).

**Decision: base is scmp_llm's `AdaptiveMPConfig`.** Port these orthogonal vit_sc features on top:

1. `fixed_levels: dict[(block_idx, op), int]` — pin a block to a specific precision level (escape hatch used by `mp_auto_calibrator`). Easy add to `AdaptiveMPConfig`.
2. `_CURRENT_BLOCK_IDX` global + `set_current_block_idx` / `get_current_block_idx` setters — vit_sc's forward-pre-hook plumbing. `adaptive_classify_rows` already takes `block_idx` as an explicit kwarg (preferred); add a fallback `block_idx = block_idx if block_idx is not None else _CURRENT_BLOCK_IDX`.
3. `AutoMPBudgetLogger` — pure log/budget bookkeeping. Copy as-is.
4. `_classify_all_rows_to_level` — helper for the `fixed_levels` path. Copy as-is.

Then `FreeBoundaryMPConfig` becomes a **thin convenience subclass** that:

- Defaults `timestep_buckets = 1`, `layer_buckets = num_blocks`.
- Exposes `set_boundaries(op, block_idx, ...)` / `set_fixed_level(...)` writing through to `bucket_thresholds` / `fixed_levels`.

Result: scmp_llm callers and vit_sc callers both compile unchanged; the underlying storage is the single `AdaptiveMPConfig` schema; ViT users still get an ergonomic per-block API.

### mp_auto_calibrator.py

vit_sc only. ~27 KB. Provides `RidgeFitter`, `auto_calibrate_mp`, `AutoMPBudgetLogger` — used by `sc_integration/sc_linear.py` and `sc_integration/mp_search.py`. scmp_llm has no equivalent today; including it in `scmp_kernels` lets scmp_llm pick up auto-calibration when it migrates. **Decision**: copy as-is.

### sng.py

Symbols identical; scmp_llm's copy is ~4 hours newer with whitespace-level differences. **Decision**: take scmp_llm.

### rng.py / lfsr_taps.py

Byte-identical. **Decision**: copy either.

### sc.py / dse.py

Byte-identical, and **both confirmed dead** in both repos (no callers). **Decision**: do not migrate. Delete from `scmp_llm/SC/` and `vit_sc/sc/` after migration lands.

---

## Recommended `scmp_kernels` layout (SC ⊥ MP)

The audit confirmed `sc_triton.py` / `config_helpers.py` never import from `mp_config.py`, and `mp_auto_calibrator.py` never imports from `sc_triton.py`. So the two concerns can — and should — live in sibling sub-packages:

```
scmp_kernels/
├── pyproject.toml
├── README.md
├── src/scmp_kernels/
│   ├── __init__.py                     # re-exports public API from sc/ and mp/
│   ├── sc/                             # stochastic computing kernels
│   │   ├── __init__.py                 # exposes sc_matmul, sc_matmul_grouped, sc_matmul_enable_*, make_sobol_simple_config
│   │   ├── sc_triton.py                # MERGED scmp_llm + vit_sc
│   │   ├── config_helpers.py           # MERGED — Sobol config builders
│   │   ├── sng.py                      # RNGPool, SNGBank (internal)
│   │   ├── rng.py                      # Sobol, LFSR (internal)
│   │   ├── lfsr_taps.py                # tap tables (internal)
│   │   └── constants.py                # FP8_E4M3_MAX, FP8_E5M2_MAX, INT8_MAX (replaces sc.py)
│   └── mp/                             # mixed-precision dispatch
│       ├── __init__.py                 # exposes classify_rows_by_metric, AdaptiveMPConfig, FreeBoundaryMPConfig, RidgeFitter, AutoMPBudgetLogger, set_current_block_idx
│       ├── config.py                   # MERGED — bucketed (diffusion) + free-boundary (ViT)
│       └── auto_calibrator.py          # from vit_sc
└── tests/
    ├── test_sc_public_api.py           # smoke-import every sc/ symbol
    └── test_mp_public_api.py           # smoke-import every mp/ symbol
```

Two side-effects of moving into a real package — both mechanical:

1. **Relative imports.** Today `sc_triton.py` does `from sng import RNGPool, SNGBank` / `from sc import FP8_E4M3_MAX, ...` — only works because both repos put `SC/` (or `sc/`) on `sys.path`. Inside `scmp_kernels.sc` these become `from .sng import RNGPool, SNGBank` / `from .constants import FP8_E4M3_MAX, ...`. Same for `config_helpers.py`'s `from sng import ...` and `from lfsr_taps import ...`.
2. **`_CURRENT_BLOCK_IDX` lives in `mp/config.py`**. The forward-pre-hook setter (`set_current_block_idx`) needs to be re-exported by `mp/__init__.py` because vit_sc's auto-calibrator installs hooks that mutate it.

`scmp_kernels/__init__.py` should expose **only** the public API in the TL;DR — anything else is internal. Downstream code that today reaches into internals (e.g. `sc_integration/sc_matmul.py` importing `RNGPool`, `SNGBank`) should be refactored to use the public entrypoints, or those internals get explicitly promoted to the public list.

---

## Migration steps (suggested order)

1. **Confirm `mp_config.py` direction** with me — that's the only design question that needs a human call. Everything else is mechanical.
2. Land `scmp_kernels` v0.1 with the merged files above and a `tests/test_public_api.py` that just imports every public symbol.
3. In `vit_sc`, replace `sc/` with `import scmp_kernels as sc` (or an `sc/__init__.py` shim that re-exports). Run `cls/` and `det/` smoke tests.
4. In `scmp_llm`, replace `SC/` the same way. Run `Q-DiT/qdit/sc_integration/` smoke tests + the kept bench/compare scripts (move them to `scmp_llm/experiments/`).
5. After both downstream repos pass, delete the dropped files from both repos and remove the now-dead in-repo `SC/` / `sc/` directories.

---

## Open questions for you

1. ~~mp_config.py direction~~ — resolved: keep both `AdaptiveMPConfig` (bucketed, for diffusion) and `FreeBoundaryMPConfig` (per-block, for ViT) since they target different model families.
2. **`fused_quantize_bipolar/unipolar`** — used by `scmp_llm/calibrate_mp_thresholds.py` but not by vit_sc. Is `calibrate_mp_thresholds.py` part of the "core" or a one-off tool? If one-off, we can keep these helpers private to scmp_llm.
3. **Bench/compare scripts** in `scmp_llm/SC/` — keep any of them? `bench_table_vs_compact.py` and `test_kernel_opt.py` reference live kernels and could become regression benchmarks. `compare_cbsg.py`, `compare_matmul.py` import nothing from SC and look obsolete.
