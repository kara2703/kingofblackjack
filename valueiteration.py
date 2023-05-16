import gymnasium as gym
from collections import defaultdict
import numpy as np
# import matplotlib.pyplot as plt
from blackjack import BlackJack
import copy
from gymnasium import Env
from baseline_policies import StickerPolicy
from tqdm import tqdm

# Win rate: 0.45062
# Draw Rate: 0.08813
# Loss Rate: 0.46125
# AverageRet: -0.01063


class ValueIteration:

    def __init__(self, env: Env, discount=1.0, theta=1e-5):
        self.env = env
        self.discount = discount
        self.theta = theta

        self.blackjack = BlackJack(StickerPolicy())

        # A dictionary from (state) -> V(state)
        self.V = defaultdict(lambda: 0)
        # A dictionary from (state) -> action maximizing the max in V(s)
        self.argV = defaultdict(lambda: 0)

        self.training_error = []

        print("Training.")
        self.train()

    # Returns a tuple (newV, argNewV)

    def _valueUpdate(self, state, simulations=100):
        # Perform a simulation to approximate the update rule:
        # V(s) = max_{a \in A(s)} \sum_{s', r} p(s', r | s, a)(r + \gamma V(s'))
        # We approximate \sum_{s', r} p(s', r | s, a)(r + \gamma V(s')) for hit
        # and stick

        # Blackjack engine for help

        # First determine hit results
        # Need to hit once then determine results
        totalHitReward = 0
        for i in range(simulations):
            # Ok sorry this is from blackjack.py but there is no good code requirement...
            (player_value, dealer_value, usable_ace) = state
            card = self.blackjack.giveCard()
            if card == 1:
                if player_value <= 10:
                    player_value += 11
                    usable_ace = True
                else:
                    player_value += 1
            else:
                player_value += card
            if player_value > 21:
                if usable_ace:
                    player_value -= 10
                    usable_ace = False

            if player_value > 21:
                # Since r = -1, and there is no next state
                totalHitReward -= 1
            else:
                totalHitReward += self.discount * \
                    self.V[(player_value, dealer_value, usable_ace)]

        totalStickReward = 0
        # Now do the stick policy
        for i in range(simulations):
            (player_value, dealer_value, usable_ace) = state
            totalStickReward += self.blackjack.play(state)

        totalHitReward /= simulations
        totalStickReward /= simulations

        return (max([totalStickReward, totalHitReward]), int(
            np.argmax([totalStickReward, totalHitReward])))

    def train(self, max_iterations=1000):
        for i in tqdm(range(max_iterations)):
            delta = 0
            for player_sum in range(self.env.observation_space[0].n):
                for dealer_card in range(self.env.observation_space[1].n):
                    for usable_ace in range(self.env.observation_space[2].n):
                        # Convert to bool for consistency
                        usable_ace = usable_ace == 1

                        state = (player_sum, dealer_card, usable_ace)

                        v = self.V[(player_sum, dealer_card, usable_ace)]

                        (newV, newArgV) = self._valueUpdate(state)

                        self.V[(player_sum, dealer_card,
                               usable_ace)] = newV
                        self.argV[(player_sum, dealer_card,
                                   usable_ace)] = newArgV

                        delta = max(delta, abs(
                            v - self.V[player_sum, dealer_card, usable_ace]))
            if delta < self.theta:
                print("BREAKKKKKKKKKK")
                break
            self.training_error.append(delta)

        print("No break!!!!!!!!!!!!!!!")

    # Value Iteration:  Compute the optimal policy
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        return self.argV[obs]
