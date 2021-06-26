import numpy as np
import matplotlib.pyplot as plt

def plot_results(rewards, path="plot.png", save=False):
    """Plot the evolution of the results (cf: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)"""

    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards)

    if len(rewards) >= 100:
        # means = [[np.mean(rewards[i - 100 :i])] * 100 for i in range(100, len(rewards), 100)]
        # means = [val for sublist in means for val in sublist]

        means = [-200] * 100 + [np.mean(rewards[i - 100 :i]) for i in range(100, len(rewards), 1)]

        if not save:
            plt.plot(means)

    plt.pause(0.001)

    if save:
        plt.savefig(path)
