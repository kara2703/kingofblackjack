
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

from gymTraining import trainGymRlModel
import evaluation as eval

from collections.abc import Mapping

from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory


class DeepQLearningSKBlackjack:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        self.n_action = env.action_space.n
        # self.n_state = env.observation_space
        self.q_network = self.gen_q_network()

        self.q_values = {}
        self.training_error = []

    def gen_q_network(self):

        # model = Sequential()
        # model.add(Input(shape=(3,), name="input"))
        # # model.add(Dense(3, , activation="relu", name="input2"))
        # model.add(Dense(units=16, activation='relu', name="dense1"))
        # model.add(Dense(units=16, activation='relu', name="dense2"))
        # model.add(Dense(self.n_action, activation='softmax', name="output"))

        inputs = Input(shape=(3, ))
        hidden1 = Dense(units=16, activation='relu')(inputs)
        hidden2 = Dense(units=16, activation='relu')(hidden1)
        outputs = Dense(units=2, activation='relu')(hidden2)

        model = Model(inputs, outputs)

        model.summary()

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
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_network.predict(self._wrap(obs), verbose=0)[0]))

        # Maybe call self.update() here?

    def _wrap(self, obs):
        return tf.constant([[obs[0], obs[1], obs[2]]])

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        obs = self._wrap(obs)
        next_obs = self._wrap(next_obs)

        # print(obs.shape)
        # print(next_obs.shape)

        # self.q_network(np.array([1, 2, 3, 4, 5, 6, 76]))

        prediciton = self.q_network(obs)[0]
        futurePrediction = self.q_network(next_obs)[0]

        future_q_value = (not terminated) * np.max(futurePrediction)

        newQValue = reward + self.discount_factor * future_q_value

        target = np.zeroes((2,))

        toUpdate = np.argmax(futurePrediction)
        target[toUpdate] = newQValue
        target[1 - toUpdate] = prediction[]

        self.q_network.fit(obs, target, verbose=0)

        temporal_difference = target - prediciton[action]
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,
                           self.epsilon - self.epsilon_decay)

    def gen_q_table(self):
        for pv in range(32):
            for ds in range(12):
                for a in range(2):
                    obs = (pv, ds, a)
                    dict[obs] = self.q_network.predict(obs)[0]

        return dict
