detect_mode_collapse <- function(generated_samples, threshold = 0.1) {
  feature_stds <- apply(generated_samples, 2, sd)
  diversity_score <- mean(feature_stds)
  is_collapsed <- diversity_score < threshold
  list(diversity_score = diversity_score, is_collapsed = is_collapsed)
}
