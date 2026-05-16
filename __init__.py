"""
thai_bias — Thai gender bias detection in multilingual embeddings
"""
from .config    import DEVICE, BATCH_SIZE, setup_matplotlib_thai, vram_info
from .embedders import EmbedderFactory, EmbedConfig, MODEL_REGISTRY
from .metrics   import run_seat, gender_axis, project, direct_bias
from .pipeline  import ModelResult, run_model, plot_model
from .analysis  import plot_within_group, plot_cross_group, final_ranking

__version__ = "0.1.0"
__all__ = [
    # config
    "DEVICE", "BATCH_SIZE", "setup_matplotlib_thai", "vram_info",
    # embedders
    "EmbedderFactory", "EmbedConfig", "MODEL_REGISTRY",
    # metrics
    "run_seat", "gender_axis", "project", "direct_bias",
    # pipeline
    "ModelResult", "run_model", "plot_model",
    # analysis
    "plot_within_group", "plot_cross_group", "final_ranking",
]
