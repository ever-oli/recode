import torch


def compute_advantage(states, rewards, V, gamma):
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)

    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        advantages[t] = G - V[states[t]]

    return advantages
