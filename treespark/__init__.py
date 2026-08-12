"""TreeSpark: calibrated, load-adaptive draft trees for semi-autoregressive
speculative decoding."""

from treespark.dspark import DSparkDraft, dspark_context_features
from treespark.tree import expand, generate, log_phat

__all__ = [
    "DSparkDraft",
    "dspark_context_features",
    "expand",
    "generate",
    "log_phat",
]
__version__ = "0.1.0"
