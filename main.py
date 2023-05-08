import evaluation as eval
import createGraphics as cg

from baseline_policies import HitterPolicy, StickerPolicy, RandomPolicy, DealerPolicy, ThorpStrategy
from montecarlo import MonteCarloESBlackjack
from gymTraining import trainGymRlModel
import gymnasium as gym


def runMonteCarloES():
    n_episodes = 100_000
    env = gym.make("Blackjack-v1", sab=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    agent = MonteCarloESBlackjack(env)

    print("Training")

    trainGymRlModel(agent, env, 10_000)

    policy = eval.policyOfGymAgent(agent)

    cg.createPolicyEvaluation(policy)
    cg.createPolicyGrid(policy)


def runThorp():
    policy = ThorpStrategy()

    (wins, draws, losses, averageRet, iters) = eval.evaluate(policy, 100000)

    print("Win rate: {}\nDraw Rate: {}\nLoss Rate: {}\nAverageRet: {}".format(
        wins / iters, draws / iters, losses / iters, averageRet))

    cg.createPolicyGrid(policy)


# runThorp()
runMonteCarloES()
