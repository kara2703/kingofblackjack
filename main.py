import evaluation as eval
import createGraphics as cg

from baseline_policies import HitterPolicy, StickerPolicy, RandomPolicy, DealerPolicy, ThorpStrategy, QPolicy
from montecarlo import MonteCarloBlackjack
from valueiteration import ValueIteration
from qlearning import QLearningBlackjack
from gymTraining import trainGymRlModel
import gymnasium as gym
from valueiteration import ValueIteration
from deep_qlearning_experiment import DeepQLearningExperBlackjack
import tensorflow as tf
import numpy as np

# Win rate: 0.437
# Draw Rate: 0.084
# Loss Rate: 0.479
# AverageRet: -0.042


def runMonteCarloES():
    n_episodes = 500_000
    env = gym.make("Blackjack-v1", sab=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

    start_epsilon = 1
    final_epsilon = 0.0
    epsilon_change = (start_epsilon - final_epsilon) / (n_episodes / 2)

    agent = MonteCarloBlackjack(
        env, start_epsilon=start_epsilon, final_epsilon=final_epsilon, epsilon_change=epsilon_change)

    print("Training")

    trainGymRlModel(agent, env, n_episodes)

    policy = eval.policyOfGymAgent(agent)

    err = np.array(agent.training_error)
    np.savetxt("OUT_MONTECARLO_V1.txt", err)

    cg.createPolicyEvaluation(policy)
    cg.createPolicyGrid(policy)

    cg.createTrainingErrorPlot(agent)
    cg.createEpisodeTrainingGraphs(env)


def runQLearning():
    n_episodes = 100_000
    # n_episodes = 1000
    env = gym.make("Blackjack-v1", sab=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

    learning_rate = 0.01
    start_epsilon = 1.0
    # reduce the exploration over time
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.0

    agent = QLearningBlackjack(
        env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        start_epsilon=start_epsilon,
        final_epsilon=final_epsilon,
        n_episodes=n_episodes
    )

    print("Training")

    trainGymRlModel(agent, env, n_episodes)

    policy = eval.policyOfGymAgent(agent)

    cg.createPolicyEvaluation(policy)
    cg.createPolicyGrid(policy)
    cg.createQValuePlot(agent)

    cg.createTrainingErrorPlot(agent)
    # cg.createEpisodeTrainingGraphs(env)


def runThorp():
    policy = ThorpStrategy()

    (wins, draws, losses, averageRet, iters) = eval.evaluate(policy, 100000)

    print("Win rate: {}\nDraw Rate: {}\nLoss Rate: {}\nAverageRet: {}".format(
        wins / iters, draws / iters, losses / iters, averageRet))

    cg.createPolicyGrid(policy)


def runValueIteration():
    n_episodes = 500_000
    env = gym.make("Blackjack-v1", sab=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

    agent = ValueIteration(env)

    policy = eval.policyOfGymAgent(agent)
    cg.createPolicyEvaluation(policy, 10000)

    cg.createPolicyGrid(policy)

    # agent.gen_q_table()

    err = np.array(agent.training_error)
    np.savetxt("OUT_VALUEITER_V1.txt", err)

    cg.createTrainingErrorPlot(agent, 1)
    # cg.createEpisodeTrainingGraphs(env)


def runDeepQlearnExper():
    # tf.random.set_seed(1865)

    n_episodes = 10000

    env = gym.make("Blackjack-v1", sab=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

    learning_rate = 0.01

    start_epsilon = 1.0
    # reduce the exploration over time
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.0

    agent = DeepQLearningExperBlackjack(
        env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        min_sample=100,
        memory_size=200,
        sample_size=75,
        discount_factor=1.0
    )

    trainGymRlModel(agent, env, n_episodes, progress=True)

    print("Finished training.  Running evaluation")

    policy = eval.policyOfGymAgent(agent)

    # qPolicy = QPolicy(agent.gen_q_table())

    # cg.createPolicyEvaluation(qPolicy, 10000)

    print("Normal policy:")
    cg.createPolicyEvaluation(policy, 1000)

    cg.createPolicyGrid(policy)

    # agent.gen_q_table()

    cg.createTrainingErrorPlot(agent)
    cg.createEpisodeTrainingGraphs(env)


# runValueIteration()
# runThorp()
runMonteCarloES()
# runQLearning()
# runDeepQlearnExper()
