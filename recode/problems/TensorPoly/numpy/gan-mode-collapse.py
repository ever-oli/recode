import numpy as np


def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    feature_stds = np.std(generated_samples, axis=0)
    diversity_score = float(np.mean(feature_stds))
    is_collapsed = diversity_score < threshold
    return {
        "diversity_score": diversity_score,
        "is_collapsed": is_collapsed,
    }
