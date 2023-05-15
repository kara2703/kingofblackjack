
import createGraphics as cg
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

from gymTraining import trainGymRlModel
import evaluation as eval

from collections.abc import Mapping

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory

n_episodes = 1000
# n_episodes = 50


env = gym.make("Blackjack-v1", sab=True)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)


class DeepQLearningBlackjack:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.n_action = env.action_space.n
        # self.n_state = env.observation_space
        self.q_network = self.gen_q_network()

        self.q_values = {}
        self.training_error = []

    def gen_q_network(self):
        model = Sequential()
        model.add(tf.keras.layers.Input(shape=(1,)))
        model.add(Dense(32, input_shape=(2,), activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.n_action, activation='softmax'))
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam())

        return model

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
            return int(np.argmax(self.q_network.predict(np.array(obs), verbose=0)[0]))

        # Maybe call self.update() here?

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        obs = np.array(obs)
        next_obs = np.array(next_obs)
        prediciton = self.q_network.predict(obs, verbose=0)

        future_q_value = (not terminated) * \
            self.q_network.predict(next_obs, verbose=0)

        target = reward + self.discount_factor * future_q_value

        temporal_difference = (np.argmax(target) - prediciton[0])

        self.q_network.fit(obs, target, verbose=0)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)

    def gen_q_table(self):
        for pv in range(32):
            for ds in range(12):
                for a in range(2):
                    obs = (pv, ds, a)
                    dict[obs] = self.q_network.predict(obs)[0]

        return dict


learning_rate = 0.1
start_epsilon = 1.0
# reduce the exploration over time
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

agent = DeepQLearningBlackjack(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Train using
trainGymRlModel(agent, env, n_episodes)

print("Finished training.  Running evaluation")

policy = eval.policyOfGymAgent(agent)


cg.createPolicyEvaluation(policy, 1000)


cg.createPolicyGrid(policy)

# (wins, draws, losses, averageRet, iters) = eval.evaluate(
#     eval.policyOfGymAgent(agent), 2000)

# print("Win rate: {}\nDraw Rate: {}\nLoss Rate: {}\nAverageRet: {}".format(
#     wins / iters, draws / iters, losses / iters, averageRet))

# eval.rlTrainingPlots(env, agent)

# # state values & policy with usable ace (ace counts as 11)
# agent.gen_q_table()
# value_grid, policy_grid = eval.create_grids(agent, usable_ace=False)
# fig1 = eval.create_plots(value_grid, policy_grid, title="Without usable ace")
# plt.show()
