# Abstract implementation of a blackjack player policy
import numpy as np


class BlackJack(object):

    """
    As our goal is to estimate the states value under our fixed policy, 
    we will need to have a (state, value) dict to record all the states 
    and number of winnings along our game simulation, and also a player_states to track the states of each game.
    """

    def __init__(self):
        self.player_state_value = {}
        self.player_states = []
        self.player_win = 0
        self.player_draw = 0

    # give card
    @staticmethod
    def giveCard():
        # 1 stands for ace
        c_list = list(range(1, 11)) + [10, 10, 10]
        return np.random.choice(c_list)

    def dealerPolicy(self, current_value, usable_ace, is_end):
        if current_value > 21:
            if usable_ace:
                current_value -= 10
                usable_ace = False
            else:
                return current_value, usable_ace, True
        # HIT17
        if current_value >= 17:
            return current_value, usable_ace, True
        else:
            card = self.giveCard()
            if card == 1:
                if current_value <= 10:
                    return current_value + 11, True, False
                return current_value + 1, usable_ace, False
            else:
                return current_value + card, usable_ace, False

    # one can only has 1 usable ace
    def playerPolicy_OLD(self, current_value, usable_ace, is_end):
        if current_value > 21:
            if usable_ace:
                current_value -= 10
                usable_ace = False
            else:
                return current_value, usable_ace, True
        # HIT17
        if current_value >= 20:
            return current_value, usable_ace, True
        else:
            card = self.giveCard()
            if card == 1:
                if current_value <= 10:
                    return current_value + 11, True, False
                return current_value + 1, usable_ace, False
            else:
                return current_value + card, usable_ace, False
    
    def playerPolicy(self, current_value, usable_ace) -> bool:
        raise "Not implemented"

    def _giveCredit(self, player_value, dealer_value, is_end=True):
        if is_end:
            # give reward only to last state
            last_state = self.player_states[-1]
            if player_value > 21:
                if dealer_value > 21:
                    # draw
                    self.player_draw += 1
                else:
                    self.player_state_value[last_state] -= 1
            else:
                if dealer_value > 21:
                    self.player_state_value[last_state] += 1
                    self.player_win += 1
                else:
                    if player_value < dealer_value:
                        self.player_state_value[last_state] -= 1
                    elif player_value > dealer_value:
                        self.player_state_value[last_state] += 1
                        self.player_win += 1
                    else:
                        # draw
                        self.player_draw += 1

    def play(self, rounds=1000):
        for i in range(rounds):
            if i % 1000 == 0:
                print("round", i)
            # hit 2 cards each
            dealer_value, player_value = 0, 0
            show_card = 0

            # give dealer 2 cards and show 1
            dealer_value += self.giveCard()
            show_card = dealer_value
            dealer_value += self.giveCard()

            # player's turn
            # always hit if less than 12
            usable_ace, is_end = False, False
            while True:
                player_value, usable_ace, is_end = self.playerPolicy(player_value, usable_ace, is_end)

                if is_end:
                    break
                # when value goes higher than 12, record states
                if (player_value >= 12) and (player_value <= 21):
                    self.player_states.append((player_value, show_card, usable_ace))
            # print("player card sum", player_value)

            # dealer's turn
            usable_ace, is_end = False, False
            while not is_end:
                dealer_value, usable_ace, is_end = self.dealerPolicy(dealer_value, usable_ace, is_end)
            # print("dealer card sum", dealer_value)

            # judge winner
            # set intermediate state to 0
            for s in self.player_states:
                self.player_state_value[s] = 0 if self.player_state_value.get(
                    s) is None else self.player_state_value.get(s)

            self._giveCredit(player_value, dealer_value)

# class BlackJackState(object):
#     """
#     BlackJackState 
#     """

class Policy(object):
    def determineAction(current_value, usable_ace, dealer_showing) -> bool:
        """
        evaluatePolicy determines whether to hit (True) or stick (False)
        based on the current blackjackState.  

        current_value is the current total of the cards
        usable_ace is true if there is an ace that can have its value reduced
        dealer_showing is a value between 1 and 10 showing the value of the 
            dealer's card, 1 for ace, 10 for 10 or face card.
        """
        