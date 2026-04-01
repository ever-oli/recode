import mlx.core as mx


def compute_advantage(states, rewards, V, gamma):
    T = len(rewards)
    advantages = mx.zeros((T,), dtype=mx.float32)

    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        advantages = mx.array(advantages)
        advantages[t] = G - V[states[t]]

    return advantages
