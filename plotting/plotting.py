import matplotlib.pyplot as plt
import torch

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