import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import DreamerV2_Agent


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: DreamerV2 for Minatar.")
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="dreamerv2_configs/dreamerv2_minatar.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = DreamerV2_Agent(config=configs, envs=envs)

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}

    Agent.finish()
