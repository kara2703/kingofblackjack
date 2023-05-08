# Here we invoke evaluation.py on the different models
import evaluation as eval


def createPolicyGrid(policy):
    """
        Display the action the given policy would take for each combination
        of input parameters
    """
    eval.createPolicy(policy, False, "No Ace").show()
    eval.createPolicy(policy, True, "Ace").show()


# Requires agent to have q_values defined
def createQValuePlot(agent):
    """
        Display a 3d plot of Q values for each combination of input parameters
    """
    eval.createValuePlot(agent, False, "No Ace").show()
    eval.createValuePlot(agent, True, "Ace").show()


def createPolicyEvaluation(policy):
    """
        Display win rate data about the given policy
    """
    (wins, draws, losses, averageRet, iters) = eval.evaluate(policy, 100000)

    print("Win rate: {}\nDraw Rate: {}\nLoss Rate: {}\nAverageRet: {}".format(
        wins / iters, draws / iters, losses / iters, averageRet))


# Create a training graph of a gymnasium agent, must
def createEpisodeTrainingGraphs(env):
    """
        Create episode training graphs, information about rewards and length
    """
    eval.episodeRewardsPlot(env).show()
    eval.episodeLengthPlot(env).show()


def createTrainingErrorPlot(agent):
    """
        Create training error plots based on agent provided data.
    """
    eval.trainingErrorPlot(agent).show()
