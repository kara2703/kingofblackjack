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
import gymnasium as gym
import numpy as np

class ValueIteration:
    def __init__(self, env, gamma=1.0, theta=1e-5):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(self.env.observation_space.n)
# self.env.observation_space is actually a tuple object that contains
# the minimum and maximum values for each observation dimension. To get the total number
#  of possible states in the observation space, you can use the n attribute of the space object
    def train(self, num_iterations=1000):
        for i in range(num_iterations):
            delta = 0
            for state in range(self.env.observation_space.n):
                v = self.V[state]
                self.V[state] = self._compute_action_value(state)
                delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break

    def _compute_action_value(self, state):
        action_values = np.zeros(self.env.action_space.n)
        for action in range(self.env.action_space.n):
            for prob, next_state, reward, done in self.env.P[state][action]:
                action_values[action] += prob * (reward + self.gamma * self.V[next_state])
        return np.max(action_values)

    def get_policy(self):
        policy = np.zeros(self.env.observation_space.n, dtype=int)
        for state in range(self.env.observation_space.n):
            action_values = np.zeros(self.env.action_space.n)
            for action in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[state][action]:
                    action_values[action] += prob * (reward + self.gamma * self.V[next_state])
            policy[state] = np.argmax(action_values)
        return policy

    def get_value_function(self):
        return self.V


def simulate_game(env, policy):
    state = env.reset()
    while True:
        action = policy[state]
        next_state, reward, done, _ = env.step(action)
        if done:
            return reward
        state = next_state


if __name__ == '__main__':
    env = gym.make('Blackjack')
    vi = ValueIteration(env)
    vi.train()
    policy = vi.get_policy()
    V = vi.get_value_function()

    num_episodes = 10000
    total_reward = 0
    for i in range(num_episodes):
        reward = simulate_game(env, policy)
        total_reward += reward
    avg_reward = total_reward / num_episodes

    print("Average reward:", avg_reward)
    print("Value function:")
    print(V)