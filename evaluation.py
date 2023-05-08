# Goal of this module is to evaluate policies and create graphs of metrics.
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from matplotlib.patches import Patch
from blackjack import Policy, BlackJack
from tqdm import tqdm


class GymPolicy(Policy):
    def __init__(self, gymPolicy):
        self.gymPolicy = gymPolicy

    def determineAction(self, player_value, usable_ace, show_card) -> bool:
        asObs = (player_value, show_card, 1 if usable_ace else 0)
        action = self.gymPolicy.get_action(asObs)

        # print("pv={}, ua={},sc={} => {}".format(player_value, usable_ace, show_card, action))

        return action == 1


# Converts a gymnasium agent into our policy class for evaluation
def policyOfGymAgent(agent) -> Policy:
    return GymPolicy(agent)


def evaluate(policy: Policy, nIterations=100_000):
    print("Evaluating a policy.")
    """
    return (wins, draws, losses, averageReturn, iterations)
    """
    wins = 0
    draws = 0
    losses = 0

    blackjackGame = BlackJack(policy)

    for i in tqdm(range(nIterations)):
        # print("--------------")
        res = blackjackGame.play()
        if res == 1:
            wins += 1
            # print("Win")
        elif res == 0:
            draws += 1
            # print("Draw")
        else:
            losses += 1
            # print("Loss")

    return (wins, draws, losses, (wins - losses) / nIterations, nIterations)


# Based on https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
# Generate plots of rl training rates
rolling_length = 500


def episodeRewardsPlot(env):
    fig, ax = plt.plot()
    ax.set_title("Episode rewards")
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )

    ax.plot(range(len(reward_moving_average)), reward_moving_average)

    return plt


def episodeLengthPlot(env):
    fig, ax = plt.plot()
    ax.set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    ax.plot(range(len(length_moving_average)), length_moving_average)
    return plt


# Requires agent to have training_error array defined
def trainingErrorPlot(agent):
    fig, ax = plt.plot()
    ax.set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.training_error),
                    np.ones(rolling_length), mode="same")
        / rolling_length
    )
    ax.plot(range(len(training_error_moving_average)),
            training_error_moving_average)

    return plt


# Based on https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/


# Create a figure


def createValuePlot(agent, usable_ace, title: str):
    """Creates a plot using a value and policy grid."""
    min_player_count = 12

    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(min_player_count, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(min_player_count, 22), range(min_player_count, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax.set_title(f"State values: {title}")
    ax.set_xlabel("Player sum")
    ax.set_ylabel("Dealer showing")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Value", fontsize=14, rotation=90)
    ax.view_init(20, 220)

    return fig


# Display a blackjack policy as a grid
def createPolicy(policy: Policy, usable_ace=False, title=""):
    min_player_count = 2

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(min_player_count, 22),
        np.arange(1, 11),
    )

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy.determineAction(obs[0], usable_ace, obs[1]),
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    fig = plt.figure(figsize=plt.figaspect(0.4))

    ax = sns.heatmap(policy_grid, linewidth=0, annot=True,
                     cmap="Accent_r", cbar=False)
    ax.set_title(f"Policy: {title}")
    ax.set_xlabel("Player sum")
    ax.set_ylabel("Dealer showing")
    ax.set_xticklabels(range(min_player_count, 22))
    ax.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1))

    return plt
