
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


class DeepQLearningExperBlackjack:
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
        hidden1 = Dense(units=16, activation='softmax')(inputs)
        hidden2 = Dense(units=16, activation='softmax')(hidden1)
        outputs = Dense(units=2, activation='softmax')(hidden2)

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
        # Now obs and next obs has shape [[1, 2, 3]]
        obs = self._wrap(obs)
        next_obs = self._wrap(next_obs)

        # Future prediction has shape [[1, 2]]
        futurePrediction = self.q_network(next_obs)
        # Now has shape [1, 2]
        futurePrediction = futurePrediction[0].numpy()

        # print("--")

        # print("Obs: {} Action: {} Reward: {} Terminated: {}".format(
        #     obs, action, reward, terminated))

        # print(futurePrediction)

        # Action chosen
        future_q_value = (not terminated) * np.max(futurePrediction)
        future_action = np.argmax(futurePrediction)

        # Calculate the new q value
        newQValue = reward + self.discount_factor * future_q_value

        # print("New Q Value: {}".format(newQValue))

        # Update new q value
        futurePrediction[action] = newQValue

        # print(futurePrediction)

        # Target has shape [[1, 2]]
        target = np.array([futurePrediction])

        self.q_network.fit(obs, target, verbose=0)

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
