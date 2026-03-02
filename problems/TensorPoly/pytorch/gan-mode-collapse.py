import torch


def detect_mode_collapse(generated_samples: torch.Tensor, threshold: float = 0.1) -> dict:
    feature_stds = torch.std(generated_samples, dim=0)
    diversity_score = float(torch.mean(feature_stds).item())
    is_collapsed = diversity_score < threshold
    return {
        "diversity_score": diversity_score,
        "is_collapsed": is_collapsed,
    }
