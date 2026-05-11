"""Stochastic-computing kernels.

Public surface: a single ``sc_matmul`` function with a ``granularity``
parameter ("per_tensor" / "per_row" / "per_head"). All inputs/outputs
are float32; quantization happens inside the Triton kernels.

The five specialized internal kernels and the ``det_kernel_tuning``
context manager are exposed as semi-public for callers that need fine-
grained access (e.g. per-head batched QK with caller-computed ranges).
"""

from .matmul import sc_matmul

# Semi-public — context manager opting in to det-tuned tile sizes on the
# batched grouped path.
from .kernels import det_kernel_tuning

__all__ = [
    "sc_matmul",
    "det_kernel_tuning",
]
