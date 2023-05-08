
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import fix_bad_spelling

from gymTraining import trainGymRlModel
import evaluation as eval

n_episodes = 100_000

env = gym.make("Blackjack-v1", sab=True)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

# Implementation of ES MonteCarlo as described in Sutton Bartol


class MonteCarloBlackjack:
    def __init__(
        self,
    ):
        # Table holding the q values of the current policy
        # qValues[obs] = [Q(obs, act) for all act]
        # So qValues[obs, act] = Q(obs, act),
        # equivalently qValues[obs] = [Q(obs, stick), Q(obs, hit)]
        self.q_values = defaultdict(lambda: np.zeroes(env.action_space.n))

        # Table holding average returns, same structure as q values
        # self.returns = defaultdict(lambda: np.zeroes(env.action_space.n))

        # Table holding number of appearances (use for calculating averages)
        # Same structure as q values
        self.nAppear = defaultdict(lambda: np.zeroes(env.action_space.n))

        # List of states appearing in the current episode, just a list of pair
        # (obs, act) so that we can update after state is terminated
        self.episodeStates = []

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        #
        return int(np.argmax(self.q_values[obs]))

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
        pass


agent = MonteCarloBlackjack()

print("Training")

trainGymRlModel(agent, env, 500_000)

print("Finished training.  Running evaluation")

(wins, draws, losses, averageRet, iters) = eval.evaluate(
    eval.policyOfGymAgent(agent), 100000)

print("Win rate: {}\nDraw Rate: {}\nLoss Rate: {}\nAverageRet: {}".format(
    wins / iters, draws / iters, losses / iters, averageRet))

# eval.rlTrainingPlots(env, agent)

# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = eval.create_grids(agent, usable_ace=False)
fig1 = eval.create_plots(value_grid, policy_grid, title="Without usable ace")
plt.show()

value_grid, policy_grid = eval.create_grids(agent, usable_ace=True)
fig1 = eval.create_plots(value_grid, policy_grid, title="With usable ace")
plt.show()
