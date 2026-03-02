import torch


def compute_advantage(states, rewards, V, gamma, device=None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32, device=device)

    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        advantages[t] = G - V[states[t]]

    return advantages
