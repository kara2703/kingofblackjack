import evaluation as eval

from baseline_policies import HitterPolicy, StickerPolicy, RandomPolicy, DealerPolicy, ThorpStrategy

policy = ThorpStrategy()

(wins, draws, losses, averageRet, iters) = eval.evaluate(policy, 100000)

print("Win rate: {}\nDraw Rate: {}\nLoss Rate: {}\nAverageRet: {}".format(
    wins / iters, draws / iters, losses / iters, averageRet))
