from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import evaluation as eval


# Implementation of MonteCarlo as described in Sutton Bartol
class MonteCarloBlackjack:
    def __init__(
        self,
        env,
        start_epsilon: float,
        final_epsilon: float,
        epsilon_change: float,
    ):
        # Table holding the q values of the current policy
        # qValues[obs] = [Q(obs, act) for all act]
        # So qValues[obs, act] = Q(obs, act),
        # equivalently qValues[obs] = [Q(obs, stick), Q(obs, hit)]
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        # Table holding average returns, same structure as q values
        # self.returns = defaultdict(lambda: np.zeroes(env.action_space.n))

        # Table holding number of appearances (use for calculating averages)
        # Same structure as q values
        self.nAppear = defaultdict(lambda: np.zeros(env.action_space.n))

        # List of states appearing in the current episode, just a list of pair
        # (obs, act) so that we can update after state is terminated
        self.episodeStates = []

        self.training_error = []

        # Epsilon-greedy setup:
        self.epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_change = epsilon_change

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        p = np.random.rand()

        a = int(np.argmax(self.q_values[obs]))

        # Divide by two since one action is going to match
        if p < self.epsilon / 2:
            # If we are being random we start off by
            return 1 - a
        else:
            return a

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        self.episodeStates.append((obs, action))

        # If we terminated we want to perform an update
        if terminated:
            for (obs, act) in self.episodeStates:
                self.nAppear[obs][act] += 1
                G = reward
                n = self.nAppear[obs][act]
                # Update average reward in q value for this state, no discounting used
                self.q_values[obs][act] = G / n + \
                    (n - 1) / n * self.q_values[obs][act]

            self.episodeStates = []

    # Not needed for MonteCarlo ES
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,
                           self.epsilon - self.epsilon_change)
