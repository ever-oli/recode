import mlx.core as mx


def detect_mode_collapse(generated_samples: mx.array, threshold: float = 0.1) -> dict:
    feature_stds = mx.std(generated_samples, axis=0)
    diversity_score = float(mx.mean(feature_stds).item())
    is_collapsed = diversity_score < threshold
    return {
        "diversity_score": diversity_score,
        "is_collapsed": is_collapsed,
    }
