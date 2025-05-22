import torch

def print_proportion(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"Hitting probability: {result:.4f}")
        return result
    return wrapper


def check_if_inside_barrier(state: torch.Tensor, barrier: torch.Tensor):

    lower_extremities = barrier[0]
    upper_extremities = barrier[1]

    inside = (state >= lower_extremities) & (state <= upper_extremities)
    return inside.all(dim=1)


@print_proportion
def hitting_probability(states: torch.Tensor, barrier: torch.Tensor):

    inside_barrier = check_if_inside_barrier(states, barrier)
    states_inside_barrier = inside_barrier.sum().item()
    proportion = states_inside_barrier / states.size(0)

    return proportion

