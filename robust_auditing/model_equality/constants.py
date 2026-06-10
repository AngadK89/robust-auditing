from __future__ import annotations

import random


DEFAULT_MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
