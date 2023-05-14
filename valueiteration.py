from collections import defaultdict
import numpy as np
# import matplotlib.pyplot as plt

"""
After defining the state space, action space, reward function, and transition function,
 the code defines the value iteration algorithm. The value iteration algorithm 
 iteratively updates the value function for each state, until the maximum change 
 in the value function is below a specified threshold (epsilon).

For each state, the algorithm calculates the Q-value for each action 
(hit or stand), and then updates the value function for the state with the 
maximum Q-value. The Q-value for an action is the expected reward for taking that 
action, plus the expected discounted value of the next state (i.e., the state that 
results from taking the action). The expected discounted value of the next state
 is the sum of the value function for each possible next state, weighted by the 
probability of transitioning to that state.

Once the value function has converged, the optimal policy can be derived
 by selecting the action with the highest Q-value for each state. 
 The policy is represented as an array of integers, where 0 represents 
 standing and 1 represents hitting.

The code then tests the value iteration algorithm by running it and printing
 out the optimal policy for each possible dealer card. The optimal policy is 
 printed as a table, where each row corresponds to a player hand (from 12 to 21) 
 and each column corresponds to a dealer card (from 1 to 10). 
 The entries in the table indicate whether the optimal action is to hit (H) or stand (S) for the corresponding player hand and dealer card.

Overall, the value iteration algorithm is a powerful tool 
for solving Markov Decision Processes (MDPs) such as blackjack. 
By iteratively updating the value function for each state, 
it can find the optimal policy for the MDP and provide insights into the best actions to take in each situation.
"""

import evaluation as eval

# NOTE this is based on https://gymnasium.farama.org/environments/toy_text/blackjack/

import numpy as np

# Define the state space

states = [(i, j, k) for i in range(2, 10) for j in range(1, 11) for k in [True, False]]
print(states)

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
def transition(player_total, dealer_card, usable_ace, action):
    if usable_ace:
        usable_ace_count = 1
    else:
        usable_ace_count = 0

    if player_total >= 12 and player_total <= 21:
        if usable_ace_count == 1:
            if dealer_card >= 7 or dealer_card == 1:
                return ((player_total, dealer_card, False), reward(player_total, dealer_card))
            elif dealer_card >= 2 and dealer_card <= 6:
                if action == 0:
                    dealer_total = dealer_card
                    while dealer_total < 17:
                        dealer_total += np.random.choice(range(1, 11))
                    return ((player_total, dealer_card, False), reward(player_total, dealer_total))
                else:
                    return ((player_total + np.random.choice([1, 11]), dealer_card, True), 0)
        else:
            if player_total >= 17:
                dealer_total = dealer_card
                while dealer_total < 17:
                    dealer_total += np.random.choice(range(1, 11))
                return ((player_total, dealer_card, False), reward(player_total, dealer_total))
            elif player_total >= 12 and player_total <= 16:
                if action == 0:
                    dealer_total = dealer_card
                    while dealer_total < 17:
                        dealer_total += np.random.choice(range(1, 11))
                    return ((player_total, dealer_card, False), reward(player_total, dealer_total))
                else:
                    return ((player_total + np.random.choice(range(1, 11)), dealer_card, False), 0)
    
    # Handle cases where the player has busted
    if player_total > 21:
        return ((player_total, dealer_card, False), reward(player_total, dealer_card))
    return ((player_total, dealer_card, False), reward(player_total, dealer_card))
# Define the policy function
def policy(state, action):
    if action == 0:
        print("no")
        return 1 if state[0] >= 20 else 0
    else:
        print("yes")
        return 1 if state[0] < 20 else 0

# Define the state index function
def state_index(state):
    return states.index(state)

# Define the value iteration algori
# Define the value iteration algorithm
def value_iteration(num_episodes=1000000, gamma=1.0, epsilon=0.0001):
    # Initialize the value function
    V = np.zeros(len(states))

    # Initialize the win and loss counts
    win_count = 0
    loss_count = 0
    policy = np.zeros((len(states), len(actions)))
    # Iterate over the episodes
    for episode in range(num_episodes):
        print("h1")
        # Initialize the state and action
        # state = (np.random.choice(range(2, 10)), np.random.choice(range(1, 11)), False)
        state = (3, 5, False)
        # action = np.random.choice(actions)
        action = 1
        print(action, state)

        # Iterate until the episode terminates
        while True:
            print("h2")
            # Take the action and observe the next state and reward
            print(transition(state[0], state[1], state[2], action))
            next_state, reward = transition(state[0], state[1], state[2], action)
            print("checker2")
            # Update the win and loss counts
            if reward == 1:
                win_count += 1
            elif reward == -1:
                loss_count += 1

            # Calculate the value of the next state
            next_state_value = 0
            print("checker1")
            for a in actions:
                print("checker")
                print(V[state_index(next_state)])
                print(policy(state, a))
                next_state_value += policy(state, a) * (reward + gamma * V[state_index(next_state)])

            # Update the value function
            print(next_state_value)
            V[state_index(state)] = next_state_value

            # Check if the episode has terminated
            if reward != 0:
                break

            # Update the state and action
            state = next_state
            action = np.random.choice(actions)

    # Compute the optimal policy
    print("here1")
  
    
    # for s in range(len(states)):
    #     state = states[s]
    #     best_value = float('-inf')
    #     for a in actions:
    #         action_value = 0
    #         for next_state, reward in [(transition(*state, a))]:
    #             action_value += policy(state, a) * (reward + gamma * V[state_index(next_state)])
    #         if action_value > best_value:
    #             best_action = a
    #             best_value = action_value
    #     policy[s, best_action] = 1.0

    # Plot the win and loss rates
    total_count = win_count + loss_count
    win_rate = win_count / total_count
    loss_rate = loss_count / total_count
    fig, ax = plt.subplots()
    ax.bar(['Wins', 'Losses'], [win_rate, loss_rate])
    ax.set_ylabel('Rate')
    ax.set_title('Win and Loss Rates')
    plt.show()

    return policy, V

# # Define the value iteration algorithm
# def value_iteration(gamma=1.0, epsilon=0.0001):
#     print("p2!")
#     # Initialize the value function
#     V = np.zeros(len(states))
#     print("p1!")
#     while True:
#         # Keep track of the maximum change in V
#         delta = 0

#         # Iterate over all states
#         for i, state in enumerate(states):
#             # Calculate the Q-value for each action
#             Q = np.zeros(len(actions))
#             for j, action in enumerate(actions):
#                 next_state, reward = transition(*state, action)
#                 Q[j] = reward + gamma * V[states.index(next_state)]

#             # Update the value function for the current state
#             V_new = np.max(Q)
#             delta = max(delta, np.abs(V[i] - V_new))
#             V[i] = V_new

#         # Check for convergence
#         if delta < epsilon:
#             break

#     # Compute the optimal policy
#     policy = np.zeros(len(states))
#     for i, state in enumerate(states):
#         Q = np.zeros(len(actions))
#         for j, action in enumerate(actions):
#             next_state, reward = transition(*state, action)
#             Q[j] = reward + gamma * V[states.index(next_state)]
#         policy[i] = np.argmax(Q)

#     return V, policy

# Test the value iteration algorithm

V, policy = value_iteration()

# Print the optimal policy
print("Optimal Policy:")
for i in range(10):
    print("Dealer showing", i + 1)
    print("Player hand\t", end="")
    for j in range(2, 10):
        state = (j, i + 1, False)
        action = policy[states.index(state)]
        print("H" if action == 1 else "S", end="")
    print("\n")

