import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from matplotlib.patches import Patch
from blackjack import Policy, BlackJack
from tqdm import tqdm

err = np.loadtxt("OUT_MONTECARLO_V1.txt")

err = err[:50_000]

CONV = 1000

fig, ax = plt.subplots(ncols=1)


# Same as in evaluation.py
ax.set_title("Training Error")
training_error_moving_average = np.convolve(
    err, np.ones(CONV), mode="same") / CONV

ax.plot(range(len(training_error_moving_average)),
        training_error_moving_average)

plt.show()
