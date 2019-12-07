"""
helper program to plot F1 vs. K values.
"""


import matplotlib.pyplot as plt
import numpy as np


def main():

    plot = plt.subplot()

    # F1 values for k = 1-10

    points = [0.89208, 0.95384,  0.97674, 0.97674, 0.96875,
              0.96875, 0.96875, 0.96124, 0.96124, 0.96124]

    plot.plot(points)
    plot.set_xlabel('Value of n_neighbors (k)')
    plot.set_ylabel('F1 Score')
    plot.set_title('F1 Scores for different values of n_neighbors (k)')
    plt.xticks(np.arange(10), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    plt.show()


if __name__ == '__main__':
    main()

