
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

import collections
from gymTraining import trainGymRlModel
import evaluation as eval

from collections.abc import Mapping

from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam, SGD
from keras.initializers import HeUniform

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
        min_sample: int = 300,
        sample_size: int = 200,
        memory_size: int = 500
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

        self.memory_size = memory_size

        self.sample_size = sample_size

        self.min_sample = min_sample
        self.batchObs = collections.deque(maxlen=memory_size)
        self.batchTarget = collections.deque(maxlen=memory_size)

    def gen_q_network(self):
        init = keras.initializers.HeUniform()

        # model = Sequential()
        # model.add(Input(shape=(3,), name="input"))
        # # model.add(Dense(3, , activation="relu", name="input2"))
        # model.add(Dense(units=16, activation='relu', name="dense1"))
        # model.add(Dense(units=16, activation='relu', name="dense2"))
        # model.add(Dense(self.n_action, activation='softmax', name="output"))

        inputs = Input(shape=(3, ))
        hidden1 = Dense(units=32, activation='relu',)(inputs)
        hidden2 = Dense(units=16, activation='relu',)(hidden1)
        hidden3 = Dense(units=16, activation='relu',)(hidden2)
        # hidden4 = Dense(units=32, activation='relu',
        #                 kernel_initializer=init)(hidden3)
        # hidden5 = Dense(units=32, activation='relu',
        #                 kernel_initializer=init)(hidden4)
        outputs = Dense(units=2, activation='linear')(hidden3)

        # inputs = Input(shape=(3, ))
        # hidden1 = Dense(units=200, activation='relu')(inputs)
        # hidden2 = Dense(units=64, activation='relu')(hidden1)
        # hidden3 = Dense(units=32, activation='relu')(hidden2)
        # outputs = Dense(units=2, activation='softmax')(hidden3)

        # inputs = Input(shape=(3, ))
        # hidden1 = Dense(units=200, activation='softmax')(inputs)
        # hidden2 = Dense(units=16, activation='softmax')(hidden1)
        # outputs = Dense(units=2, activation='softmax')(hidden2)

        model = Model(inputs, outputs)

        model.summary()

        model.compile(loss='mean_squared_error',
                      optimizer=SGD(learning_rate=0.001))

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
            return int(np.argmax(self.q_network(self._wrap(obs))[0]))

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
        self.batchObs.append(obs)

        # Now obs and next obs has shape [[1, 2, 3]]
        obs = self._wrap(obs)
        next_obs = self._wrap(next_obs)

        # Future prediction has shape [[1, 2]]
        futurePrediction = self.q_network(next_obs)
        # Now has shape [1, 2]
        futurePrediction = futurePrediction[0].numpy()

        # Action chosen
        future_q_value = (not terminated) * np.max(futurePrediction)

        # Calculate the new q value
        newQValue = reward + self.discount_factor * future_q_value

        # Update new q value
        futurePrediction[action] = newQValue

        # print(futurePrediction)

        # Compute current value
        currentQValues = self.q_network(obs)[0]

        self.training_error.append(newQValue - currentQValues[action])

        self.batchTarget.append(futurePrediction)

        if len(self.batchTarget) >= self.min_sample:
            allObservations = np.array(self.batchObs)
            target = np.array(self.batchTarget)

            # print(allObservations.shape)

            sample = np.random.choice(
                allObservations.shape[0], self.sample_size, replace=False)

            allObservationsSample = allObservations[sample]
            targetSample = target[sample]

            self.q_network.fit(allObservationsSample,
                               targetSample, verbose=0)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,
                           self.epsilon - self.epsilon_decay)

    def gen_q_table(self):
        print("Generating Q Table")

        retDict = defaultdict(lambda: np.zeros(2))

        toPredict = [[pv, ds, a]
                     for pv in range(32) for ds in range(12) for a in range(2)]
        toPredictNumpy = np.array(toPredict)

        # print(len(toPredict))
        # print(toPredictNumpy.shape)

        qValues = self.q_network.predict(toPredictNumpy)

        # print(qValues.shape)
        # print(qValues)

        for obs in toPredict:
            obsTuple = (obs[0], obs[1], obs[2])

            retDict[obsTuple] = qValues[toPredict.index(obs)]

        # for pv in range(32):
        #     for ds in range(12):
        #         for a in range(2):
        #             obs = (pv, ds, a)
        #             obsPred = self._wrap(obs)

        #             retDict[obs] = self.q_network.predict(
        #                 obsPred, verbose=0)[0]

        return retDict
