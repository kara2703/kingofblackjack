# Abstract implementation of a blackjack player policy
import numpy as np


class Policy(object):
    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        """
        evaluatePolicy determines whether to hit (True) or stick (False)
        based on the current blackjackState.  

        retuyrns True to hit or False to stick

        current_value is the current total of the cards
        usable_ace is true if there is an ace that can have its value reduced
        dealer_showing is a value between 1 and 10 showing the value of the 
            dealer's card, 1 for ace, 10 for 10 or face card.
        """
        pass


class BlackJack(object):

    """
    As our goal is to estimate the states value under our fixed policy, 
    we will need to have a (state, value) dict to record all the states 
    and number of winnings along our game simulation, and also a player_states to track the states of each game.
    """

    def __init__(self, playerPolicy: Policy):
        self.player_state_value = {}
        self.player_states = []
        self.player_win = 0
        self.player_draw = 0

        self.playerPolicy = playerPolicy

    # give card
    @staticmethod
    def giveCard():
        # 1 stands for ace
        c_list = list(range(1, 11)) + [10, 10, 10]
        return np.random.choice(c_list)

    def _dealerPolicy(self, current_value, usable_ace, is_end):
        if current_value > 21:
            if usable_ace:
                current_value -= 10
                usable_ace = False
            else:
                return current_value, usable_ace, True
        # Stick on for dealer 17
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

    # Return 1 if player wins, 0 if draw, -1 if loses
    def _determineWinner(self, playerValue, dealerValue):
        # Player loses if they bust since they go first, so check first
        if playerValue > 21:
            return -1
        elif dealerValue > 21:
            return 1
        else:
            if playerValue > dealerValue:
                return 1
            elif playerValue == dealerValue:
                return 0
            else:
                return -1

    def play(self):
        # hit 2 cards each
        dealer_value, player_value = 0, 0
        show_card = 0

        # give dealer 2 cards and show 1
        dealer_value += self.giveCard()
        show_card = dealer_value
        dealer_value += self.giveCard()

        usable_ace = False

        # Give player two cards, set up usable ace
        NUM_PLAYER_CARDS = 2
        for i in range(NUM_PLAYER_CARDS):
            card = self.giveCard()

            if card == 1:
                if usable_ace:
                    player_value += 1
                else:
                    player_value += 11
                    usable_ace = True
            else:
                player_value += card

        player_done = False
        while not player_done:
            playerHit = self.playerPolicy.determineAction(
                player_value, usable_ace, show_card)

            # Determine hit action
            if playerHit:
                # Player has decided to hit
                card = self.giveCard()
                if card == 1:
                    if player_value <= 10:
                        # Here we got an ace, but our current value is less than 10 so use it as an 11 and
                        # give a usable ace
                        player_value += 11
                        usable_ace = True
                    else:
                        # Otherwise we keep the current usable ace and add one
                        player_value += 1
                else:
                    # If its not an ace, just add the value of the card
                    player_value += card
            else:
                # We chose to stick, so are done.
                player_done = True

            # Handle bust case
            if player_value > 21:
                if usable_ace:
                    player_value -= 10
                    usable_ace = False
                else:
                    # We are over 21 with no aces, so done
                    player_done = True

        dealerAce, dealerDone = False, False
        while not dealerDone:
            dealer_value, dealerAce, dealerDone = self._dealerPolicy(
                dealer_value, dealerAce, dealerDone)

        return self._determineWinner(player_value, dealer_value)
