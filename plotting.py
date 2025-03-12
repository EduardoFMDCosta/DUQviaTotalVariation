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

def plot_interval(gmm_prob_set: torch.Tensor,
                  alpha_set: torch.Tensor,
                  beta_set: torch.Tensor,
                  actual_set_prob: torch.Tensor = None):

    t = torch.arange(len(gmm_prob_set))
    lbs = gmm_prob_set + alpha_set
    ubs = gmm_prob_set + beta_set

    plt.fill_between(t, lbs, ubs, color="lightgrey", label = r'$[\hat{\mathbb{P}}_{x_t}(U) + \alpha_{U, t}, \hat{\mathbb{P}}_{x_t}(U) + \beta_{U, t}]$')

    plt.plot(t, gmm_prob_set, label=r'$\hat{\mathbb{P}}_{x_t}(U)$', linestyle='-', marker='s', color='red')
    if actual_set_prob is not None:
        plt.plot(t, actual_set_prob, label=r'$\mathbb{P}_{x_t}(U)$', linestyle='-', marker='s', color='green')

    plt.xlabel("Time step")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.show()