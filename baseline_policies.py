from blackjack import Policy
import random

# Random policy, wins ~30% of the time, average return of -.26


class RandomPolicy(Policy):
    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        return True if random.randint(0, 1) == 1 else False


class StickerPolicy(Policy):
    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        return False


# Never wins, draws 30%, indicating dealer busts 30% of time
class HitterPolicy(Policy):
    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        return True


class DealerPolicy(Policy):
    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        return True if player_value < 17 else False
