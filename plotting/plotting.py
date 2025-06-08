import torch
import numpy as np
import matplotlib.pyplot as plt
from grid.regions import HyperRectangularPartition, HyperRectangle
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy.ma as ma
from matplotlib.colors import ListedColormap
from targeted_probability.bounds import TargetedBounds

plt.style.use('seaborn-v0_8-bright')

# plt.rcParams.update({
#     'font.size': 12,
#     'text.usetex': True,
#     'text.latex.preamble': r'\usepackage{amsfonts}'
# })

def plot_samples(monte_carlo_samples: torch.Tensor,
                 gmm_samples: torch.Tensor,
                 avoid_sets: HyperRectangle = None,
                 reach_sets: HyperRectangle = None,
                 save: str = None):

    if monte_carlo_samples.shape[-1] >= 2:
        assert monte_carlo_samples.shape[0] == gmm_samples.shape[0]

        T = monte_carlo_samples.shape[0]
        colors = cm.viridis(np.linspace(0, 1, T))  # color for each timestep

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        titles = ['Monte Carlo', 'GMM']
        samples_list = [monte_carlo_samples, gmm_samples]

        # Compute global bounds for consistent axis limits and binning
        all_samples = torch.cat([monte_carlo_samples, gmm_samples], dim=1).cpu().numpy()
        xmin, xmax = all_samples[:, :, 0].min(), all_samples[:, :, 0].max()
        ymin, ymax = all_samples[:, :, 1].min(), all_samples[:, :, 1].max()

        # Define bin edges (you can customize bin count)
        bins = 100
        x_edges = np.linspace(xmin, xmax, bins + 1)
        y_edges = np.linspace(ymin, ymax, bins + 1)

        cmap = cm.get_cmap("viridis", T)  # or any sequential colormap

        for ax, samples, title in zip(axes, samples_list, titles):
            for t in range(T):
                data = samples[t].cpu().numpy()
                H, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=[x_edges, y_edges])

                # Normalize for consistent transparency
                H = H.T  # transpose to match image orientation

                H_masked = ma.masked_where(H == 0, H)  # mask zero entries

                # Use imshow with alpha blending
                ax.imshow(
                    H_masked,
                    extent=[xmin, xmax, ymin, ymax],
                    origin='lower',
                    cmap=cm.viridis,
                    alpha=0.7,  # transparency per time step
                )
                color = cmap(t / (T - 1))  # returns RGBA
                ax.imshow(
                    np.ones_like(H_masked),
                    extent=[xmin, xmax, ymin, ymax],
                    origin='lower',
                    cmap=ListedColormap([color]),
                    alpha=0.5 * (H_masked > 0),
                )


            # Plot unsafe sets if provided
            if avoid_sets is not None:
                lower = avoid_sets.lower.cpu().numpy()
                upper = avoid_sets.upper.cpu().numpy()
                for l, u in zip(lower, upper):
                    width, height = u[0] - l[0], u[1] - l[1]
                    rect = patches.Rectangle((l[0], l[1]), width, height,
                                             linewidth=1.5,
                                             edgecolor='r',
                                             facecolor='red',
                                             alpha=0.4)
                    ax.add_patch(rect)

            # Plot reach sets if provided
            if reach_sets is not None:
                lower = reach_sets.lower.cpu().numpy()
                upper = reach_sets.upper.cpu().numpy()
                for l, u in zip(lower, upper):
                    width, height = u[0] - l[0], u[1] - l[1]
                    rect = patches.Rectangle((l[0], l[1]), width, height,
                                             linewidth=1.5,
                                             edgecolor='g',
                                             facecolor='green',
                                             alpha=0.4)
                    ax.add_patch(rect)

            ax.set_title(title)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_aspect('equal')

        plt.tight_layout()
        if save:
            filename = "figures/" + save + ".png"
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()

def plot_partition(partition: HyperRectangularPartition,
                   high_prob_set: HyperRectangle = None):

    inner_partition = partition.inner_partition
    lower = inner_partition.lower
    upper = inner_partition.upper
    safety_type = partition.safety_type

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
        ax.set_facecolor('lightgrey')

        unique_types = np.unique(safety_type)
        face_color = {0: 'lightgrey', -1: 'red', 1: 'green'}
        edge_color = {0: 'grey', -1: 'red', 1: 'green'}

        # Loop over rectangles
        for lo, hi, t in zip(lower, upper, safety_type):
            width = hi[0] - lo[0]
            height = hi[1] - lo[1]
            rect = patches.Rectangle(lo, width, height, linewidth=1, edgecolor=edge_color[t.item()], facecolor=face_color[t.item()], alpha=0.5)
            ax.add_patch(rect)

        if high_prob_set is not None:
            # --- Add high probability rectangles ---
            hp_lower = high_prob_set.lower
            hp_upper = high_prob_set.upper

            # Handle multiple high-probability rectangles
            for lo, hi in zip(hp_lower, hp_upper):
                width = hi[0] - lo[0]
                height = hi[1] - lo[1]
                rect = patches.Rectangle(lo, width, height, linewidth=2,
                                         edgecolor='black', facecolor='none',
                                         linestyle='-')
                ax.add_patch(rect)

        ax.set_aspect('equal')
        ax.set_xlim(lower[:, 0].min() - 0.5, upper[:, 0].max() + 0.5)
        ax.set_ylim(lower[:, 1].min() - 0.5, upper[:, 1].max() + 0.5)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")

        plt.tight_layout()
        plt.show()

def plot_partition_bounds(partition: HyperRectangularPartition,
                          bounds: TargetedBounds):

    inner_partition = partition.inner_partition
    lower = inner_partition.lower
    upper = inner_partition.upper

    lb_delta = bounds.lb_delta.detach().cpu().numpy()
    ub_delta = bounds.ub_delta.detach().cpu().numpy()
    labels = [r'$\check \Delta(\mathcal{R})$', r'$\hat \Delta(\mathcal{R})$']

    if lower.shape[-1] == 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

        norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
        cmap = cm.viridis  # Choose any matplotlib colormap

        # Loop over rectangles
        for ax, bound, title in zip(axs, [lb_delta, ub_delta], labels):
            for lo, hi, val in zip(lower, upper, bound):
                width = hi[0] - lo[0]
                height = hi[1] - lo[1]
                color = cmap(norm(val))
                rect = patches.Rectangle(lo, width, height,
                                         linewidth=1,
                                         edgecolor='black',
                                         facecolor=color,
                                         alpha=0.7)
                ax.add_patch(rect)

            ax.set_aspect('equal')
            ax.set_xlim(lower[:, 0].min() - 0.5, upper[:, 0].max() + 0.5)
            ax.set_ylim(lower[:, 1].min() - 0.5, upper[:, 1].max() + 0.5)
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_title(title)

            # Shared colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05, pad=0.02)
        cbar.set_label("Value", rotation=270, labelpad=15)

        plt.show()

def plot_confidence_interval(gmm_prob_set: torch.Tensor,
                             alpha_set: torch.Tensor,
                             beta_set: torch.Tensor,
                             actual_set_prob: torch.Tensor = None,
                             save: str = None):

    t = torch.arange(len(gmm_prob_set))
    lbs = gmm_prob_set + alpha_set
    ubs = gmm_prob_set + beta_set

    plt.plot(t, lbs, linestyle='--', marker='s', color='grey')
    plt.plot(t, ubs, linestyle='--', marker='s', color='grey')
    plt.fill_between(t, lbs, ubs, color="lightgrey", label = r'Bounds')

    plt.plot(t, gmm_prob_set, label=r'$\hat{\mathbb{P}}_{x_t}(U)$', linestyle='-', marker='s', color='red')
    if actual_set_prob is not None:
        plt.plot(t, actual_set_prob, label=r'$\mathbb{P}_{x_t}(U)$', linestyle='-', marker='s', color='green')

    plt.xlabel("Time step")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    plt.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin

    if save:
        filename = filename = "figures/" + save + ".png"
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()