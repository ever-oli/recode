import numpy as np


def compute_advantage(states, rewards, V, gamma):
    T = len(rewards)
    advantages = np.zeros(T, dtype=float)

    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        advantages[t] = G - V[states[t]]

    return advantages
