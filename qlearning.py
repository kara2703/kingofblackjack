
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym

from gymTraining import trainGymRlModel
import evaluation as eval

#NOTE this is taken from https://gymnasium.farama.org/environments/toy_text/blackjack/

n_episodes = 100_000


env = gym.make("Blackjack-v1", sab=True)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

class QLearningBlackjack:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)

learning_rate = 0.1
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = QLearningBlackjack(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Train using 
trainGymRlModel(agent, env, n_episodes)

print("Finished training.  Running evaluation")

(wins, draws, losses, averageRet, iters) = eval.evaluate(eval.policyOfGymAgent(agent), 100000)

print("Win rate: {}\nDraw Rate: {}\nLoss Rate: {}\nAverageRet: {}".format(wins / iters, draws / iters, losses / iters, averageRet))

eval.rlTrainingPlots(env, agent)

# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = eval.create_grids(agent, usable_ace=False)
fig1 = eval.create_plots(value_grid, policy_grid, title="Without usable ace")
plt.show()