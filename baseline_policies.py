from blackjack import Policy
import random
import numpy as np


class RandomPolicy(Policy):
    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        return True if random.randint(0, 1) == 1 else False


# Kind of silly, but here is the stats
# Win rate: 0.40686
# Draw Rate: 0.05042
# Loss Rate: 0.54272
# AverageRet: -0.13586
class StickerPolicy(Policy):
    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        return False


# Never wins since player goes first, giving loss rate of 1
# Win rate: 0, loss rate 1
class HitterPolicy(Policy):
    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        return True


# Win rate: 0.43102
# Draw Rate: 0.1027
# Loss Rate: 0.46628
# AverageRet: -0.03526
class DealerPolicy(Policy):
    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        return True if player_value < 17 else False


class QPolicy(Policy):
    def __init__(self, q_values):
        self.q_values = q_values

    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """

        obs = (player_value, show_card, 1 if usable_ace else 0)

        return int(np.argmax(self.q_values[obs])) == 1

# Thorp 1966 describes this strategy for playing blackjack, said to be an
# optimal "Basic" strategy.  In theory all of our learning methods should
# be attempting to approximate this strategy
# Win rate: 0.45117
# Draw Rate: 0.09274
# Loss Rate: 0.45609
# AverageRet: -0.00492
# As we can see this is very close to an optimal strategy


class ThorpStrategy(Policy):
    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        # table[yourAmount - 12][show_card - 1]
        #
        hardTable = np.array([
            # A  2  3  4  5  6  7  8  9  10
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],  # 12
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 13
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 14
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 15
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 16
        ])

        # Soft stand on 19 or greater, otherwise use table if you have 18
        # Hit on 17 or less
        # table[show_card - 1]
        softTable = np.array(
            # A  2  3  4  5  6  7  8  9  10
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 18
        )

        if usable_ace:
            if player_value <= 17:
                return True
            elif player_value >= 19:
                return False
            else:
                return True if softTable[show_card - 1] == 1 else False
        else:
            if player_value <= 11:
                return True
            elif player_value > 16:
                return False
            else:
                return True if hardTable[player_value - 12][show_card - 1] else False
