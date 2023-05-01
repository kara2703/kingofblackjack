import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# https://github.com/MJeremy2017/reinforcement-learning-implementation/blob/master/BlackJack/blackjack_mc.py
# https://github.com/MJeremy2017/reinforcement-learning-implementation/blob/master/BlackJack/blackjack_solution.py
class BlackJackMC(object):

    """
    As our goal is to estimate the states’ value under our fixed policy, 
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

    """
    dealer's policy: Our dealer policy is to hit when card value less 
    than 17 and stand when 17 or above. The function will be able to 
    return a new value based on the action chosen and 
    be able to tell if the game is ended.
    This function can be called recursively until reaches its end as 
    its returns is the same as inputs. We keep track of the usable
    ace, when current value is less than 10, the ace will always
    be taken as 11, otherwise 1; When current value is over 21
    and the dealer has usable ace on hand, then the usable ace will
    be taken as 1, total value is subtracted by 10 accordingly, and 
    the usable ace indicator will be set to false .
    """
    # def dealerPolicy(self, current_value, usable_ace, is_end):
    #     if current_value > 21:
    #         if usable_ace:
    #             current_value -= 10
    #             usable_ace = False
    #         else:
    #             return current_value, usable_ace, True
    #     # HIT17
    #     if current_value >= 17:
    #         return current_value, usable_ace, True
    #     else:
    #         card = self.giveCard()
    #         if card == 1:
    #             if current_value <= 10:
    #                 return current_value + 11, True, False
    #             return current_value + 1, usable_ace, False
    #         else:
    #             return current_value + card, usable_ace, False
class Dealer:
    def __init__(self):
        self.hand = []
        self.stand_value = 17
    
    def get_hand_value(self):
        hand_value = 0
        for card in self.hand:
            hand_value += card.get_value()
            if card.rank == "Ace" and hand_value > 21:
                hand_value -= 10
        return hand_value
    
    def take_turn(self, deck):
        while self.get_hand_value() < self.stand_value:
            self.hit(deck)
    
    def hit(self, deck):
        self.hand.append(deck.draw_card())
    
    def show_hand(self, hide_first_card=False):
        if hide_first_card:
            print("Dealer's hand: [Hidden Card, {}]".format(self.hand[1]))
        else:
            print("Dealer's hand: {}".format(self.hand))

    # one can only has 1 usable ace
    def playerPolicy(self, current_value, usable_ace, is_end):
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


if __name__ == "__main__":
    rounds = 10000
    b = BlackJackMC()
    b.play(rounds)

    print("Player wining rate", b.player_win / rounds)
    print("Not losing rate", (b.player_win + b.player_draw) / rounds)

    print("Plots ----------------")
    usable_ace = {}
    nonusable_ace = {}

    for k, v in b.player_state_value.items():
        if k[2]:
            usable_ace[k] = v
        else:
            nonusable_ace[k] = v

    fig = plt.figure(figsize=[15, 6])

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    x1 = [k[1] for k in usable_ace.keys()]
    y1 = [k[0] for k in usable_ace.keys()]
    z1 = [v for v in usable_ace.values()]
    ax1.scatter(x1, y1, z1)

    ax1.set_title("usable ace")
    ax1.set_xlabel("dealer showing")
    ax1.set_ylabel("player sum")
    ax1.set_zlabel("reward")

    x2 = [k[1] for k in nonusable_ace.keys()]
    y2 = [k[0] for k in nonusable_ace.keys()]
    z2 = [v for v in nonusable_ace.values()]
    ax2.scatter(x2, y2, z2)

    ax2.set_title("non-usable ace")
    ax2.set_xlabel("dealer showing")
    ax2.set_ylabel("player sum")
    ax2.set_zlabel("reward")

    plt.show()