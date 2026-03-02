function detect_mode_collapse(generated_samples; threshold::Float64=0.1)
    feature_stds = mapslices(std, generated_samples; dims=1)
    diversity_score = mean(feature_stds)
    is_collapsed = diversity_score < threshold
    return (diversity_score = diversity_score, is_collapsed = is_collapsed)
end
