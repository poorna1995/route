"""Set Hugging Face hub env before any hub import (RunPod XET downloads often fail)."""

from __future__ import annotations

import os

# Must be set before huggingface_hub / transformers import hub code paths.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
