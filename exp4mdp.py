# Experiment 4: Markov Decision Process (MDP)

# -----------------------------------
# 1. Import Library
# -----------------------------------
import numpy as np

print("\n Markov Decision Process \n - exp4mdp.py:8")

# -----------------------------------
# 2. Define States and Actions
# -----------------------------------
# States: 0 = Low, 1 = Medium, 2 = High
# Actions: 0 = Stay, 1 = Move

# -----------------------------------
# 3. Transition Probabilities
# -----------------------------------
P = {
    0: {0: [0.7, 0.3, 0.0], 1: [0.2, 0.6, 0.2]},
    1: {0: [0.1, 0.8, 0.1], 1: [0.0, 0.5, 0.5]},
    2: {0: [0.0, 0.3, 0.7], 1: [0.0, 0.2, 0.8]}
}

# -----------------------------------
# 4. Rewards
# -----------------------------------
R = {
    0: {0: 5, 1: 10},
    1: {0: 2, 1: 8},
    2: {0: 1, 1: 6}
}

# -----------------------------------
# 5. Value Iteration Algorithm
# -----------------------------------
gamma = 0.9   # Discount factor
V = np.zeros(3)

for i in range(10):
    new_V = np.zeros(3)
    for s in range(3):
        action_values = []
        for a in [0, 1]:
            value = R[s][a] + gamma * sum(P[s][a][s1] * V[s1] for s1 in range(3))
            action_values.append(value)
        new_V[s] = max(action_values)
    V = new_V

# -----------------------------------
# 6. Output Results
# -----------------------------------
print("Optimal State Values: - exp4mdp.py:53", V)