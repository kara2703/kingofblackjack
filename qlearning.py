import random

# Define the Q-table
Q = {}

# Define the learning rate and discount factor
alpha = 0.5
gamma = 1.0

# Define the exploration/exploitation strategy
epsilon = 0.1

# Define the initial state of the game
state = (0, 0, False) # (player sum, dealer showing card, usable ace)

# Define the actions
actions = ['hit', 'stand']

# Define the reward function
def reward(player_sum, dealer_sum):
    if player_sum > 21:
        return -1
    elif dealer_sum > 21 or player_sum > dealer_sum:
        return 1
    elif player_sum == dealer_sum:
        return 0
    else:
        return -1

# Initialize the Q-table
for player_sum in range(12, 22):
    for dealer_sum in range(1, 11):
        for usable_ace in [True, False]:
            state = (player_sum, dealer_sum, usable_ace)
            Q[state] = {}
            for action in actions:
                Q[state][action] = 0.0

# Define the function for choosing an action
def choose_action(state):
    if random.random() < epsilon:
        action = random.choice(actions)
    else:
        q_values = Q[state]
        max_q = max(q_values.values())
        actions_with_max_q = [a for a, q in q_values.items() if q == max_q]
        action = random.choice(actions_with_max_q)
    return action

# Play the game using Q-learning
for episode in range(100000):
    # Start a new game
    player_sum = random.randint(12, 21)
    dealer_sum = random.randint(1, 10)
    usable_ace = False
    if player_sum == 21:
        usable_ace = True
    state = (player_sum, dealer_sum, usable_ace)

    # Play the game until the player stands or busts
    while True:
        action = choose_action(state)
        if action == 'hit':
            card = random.randint(1, 10)
            if card == 1 and player_sum + 11 <= 21:
                player_sum += 11
                usable_ace = True
            else:
                player_sum += card
                usable_ace = False
            if player_sum > 21:
                reward_value = -1
                break
            else:
                state = (player_sum, dealer_sum, usable_ace)
        else:
            break

    # Play the dealer's turn
    if player_sum <= 21:
        while dealer_sum < 17:
            card = random.randint(1, 10)
            if card == 1 and dealer_sum + 11 <= 21:
                dealer_sum += 11
            else:
                dealer_sum += card
        reward_value = reward(player_sum, dealer_sum)

    # Update the Q-table
    for s, a in zip(states, actions):
        q_value = Q[s][a]
        next_q_values = Q[next_state]
        max_next_q = max(next_q_values.values())
        target = reward_value + gamma * max_next_q
        new_q_value = (1 - alpha) * q_value + alpha * target
        Q[s][a] = new_q_value
