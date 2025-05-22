import torch
import matplotlib.pyplot as plt
from grid.regions import HyperRectangularPartition, HyperRectangle
import matplotlib.patches as patches

plt.style.use('seaborn-v0_8-bright')

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def plot_samples(monte_carlo_samples: list,
                 gmm_samples: list):

    assert len(monte_carlo_samples) == len(gmm_samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for t in range(len(monte_carlo_samples)):

        mc = monte_carlo_samples[t]
        gmm = gmm_samples[t]

        axes[0].scatter(mc[:, 0], mc[:, 1], color='blue', alpha=0.5)
        axes[1].scatter(gmm[:, 0], gmm[:, 1], color='red', alpha=0.5)


    plt.tight_layout()
    plt.show()

def plot_partition(partition: HyperRectangularPartition):

    inner_partition = partition.inner_partition
    lower = inner_partition.lower
    upper = inner_partition.upper

    if lower.shape[-1] == 1:
        fig, ax = plt.subplots()

        for i, (lo, hi) in enumerate(zip(lower, upper)):
            ax.plot([lo.item(), hi.item()], [i, i], color='blue', marker='|')

        ax.set_yticks(range(len(lower)))
        ax.set_yticklabels([f'Interval {i}' for i in range(len(lower))])
        ax.set_xlabel("x")

        plt.tight_layout()
        plt.show()

    elif lower.shape[-1] == 2:
        fig, ax = plt.subplots()

        # Loop over rectangles
        for lo, hi in zip(lower, upper):
            width = hi[0] - lo[0]
            height = hi[1] - lo[1]
            rect = patches.Rectangle(lo, width, height, linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        ax.set_aspect('equal')
        ax.set_xlim(lower[:, 0].min() - 0.5, upper[:, 0].max() + 0.5)
        ax.set_ylim(lower[:, 1].min() - 0.5, upper[:, 1].max() + 0.5)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")

        plt.tight_layout()
        plt.show()