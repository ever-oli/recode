compute_advantage <- function(states, rewards, V, gamma) {
  T <- length(rewards)
  advantages <- numeric(T)

  G <- 0.0
  for (t in rev(seq_len(T))) {
    G <- rewards[t] + gamma * G
    advantages[t] <- G - V[states[t]]
  }

  advantages
}
