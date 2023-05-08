from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from gymTraining import trainGymRlModel
import evaluation as eval

# NOTE this is based on https://gymnasium.farama.org/environments/toy_text/blackjack/

import numpy as np

# Define the state space
states = [(i, j, k) for i in range(12, 22) for j in range(1, 11) for k in [True, False]]

# Define the action space
actions = [0, 1]  # 0 = stand, 1 = hit

# Define the reward function
def reward(player_total, dealer_total):
    if player_total > 21:
        return -1
    elif dealer_total > 21:
        return 1
    elif player_total > dealer_total:
        return 1
    elif player_total == dealer_total:
        return 0
    else:
        return -1

# Define the transition function
def transition(player_total, dealer_card, action):
    if action == 0:  # stand
        dealer_total = dealer_card
        while dealer_total < 17:
            dealer_total += np.random.choice(range(1, 11))
        r = reward(player_total, dealer_total)
        return ((player_total, dealer_card, False), r)
    else:  # hit
        player_total += np.random.choice(range(1, 11))
        if player_total > 21:
            r = reward(player_total, dealer_card)
            return ((player_total, dealer_card, False), r)
        else:
            return ((player_total, dealer_card, True), 0)

# # Define the value iteration algorithm
# def value_iteration(gamma=1.0, epsilon=0.0001):
#     # Initialize the value function
#     V = np.zeros(len(states))

#     while True:
#         # Keep track of the maximum change in V
#         delta = 0

#         # Iterate over all states
#         for i, state in enumerate(states):
#             # Calculate the Q-value for each action
#             Q = np.zeros(len(actions))
#             for j, action in enumerate(actions):
#                 next_state, reward = transition(*state, action)
#                 Q[j] = reward + gamma *