import os
import sys

# The cudaMallocAsync allocator keeps a training step's peak from fragmenting past
# 16 GB (see deai/vram.py). It must be chosen before torch loads; a process that
# imported torch first keeps the default allocator, and an explicit
# PYTORCH_CUDA_ALLOC_CONF always wins.
if "torch" not in sys.modules:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
