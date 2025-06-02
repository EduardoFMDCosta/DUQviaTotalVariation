import json
import argparse
from typing import Optional
import os

dir = os.path.dirname(os.path.abspath(__file__))

def load_json(filename: str):
    file_path = os.path.join(dir, f"{filename}.json")
    with open(file_path, "r") as read_file:
        data = json.load(read_file)
    return data


def param_handler(param_name: str, dataset_name: str, num_dims: int, setting_tag: int = None):
    params = load_json(param_name)[dataset_name]
    if "dimensions" in params:
        return params["dimensions"][str(num_dims)]["options"][str(setting_tag)]
    else:
        return params["options"][str(setting_tag)]


def parse_arguments(
        dynamics_type,
        num_dims,
        dynamics_setting,
        prediction_horizon,
        num_samples,
        plot
):
    parser = argparse.ArgumentParser(description='Setup experiments for dynamics.')
    parser.add_argument('--dynamics_type',
                        type=str,
                        choices=['LinearDynamics'],
                        default=dynamics_type,
                        help='Type of dynamics to use.')
    parser.add_argument('--num_dims',
                        type=int,
                        default=num_dims,
                        help='Number of dimensions of the dynamics')
    parser.add_argument('--dynamics_setting',
                        type=int,
                        default=dynamics_setting,
                        help='Parameters for the dynamics as a dictionary string.')
    parser.add_argument('--prediction_horizon',
                        type=int,
                        default=prediction_horizon,
                        help='Time horizon for propagation.')
    parser.add_argument('--num_samples',
                        type=int,
                        default=num_samples,
                        help='Number of samples for empirical distribution estimate.')
    parser.add_argument('--plot',
                        type=bool,
                        default=plot,
                        help='Plot the dynamics and distributions.')

    return parser.parse_args()


def load_params(args):
    system_params = param_handler(
        param_name="parameters",
        dataset_name=args.dynamics_type,
        num_dims=args.num_dims,
        setting_tag=args.dynamics_setting
    )

    return {"dynamics_type": args.dynamics_type,
            "num_samples": args.num_samples,
            "prediction_horizon": args.prediction_horizon,
            "plot": args.plot,
            **system_params}