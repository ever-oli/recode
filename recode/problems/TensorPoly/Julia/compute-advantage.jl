function compute_advantage(states, rewards, V, gamma)
    T = length(rewards)
    advantages = zeros(Float64, T)

    G = 0.0
    for t in T:-1:1
        G = rewards[t] + gamma * G
        advantages[t] = G - V[states[t]]
    end

    return advantages
end
