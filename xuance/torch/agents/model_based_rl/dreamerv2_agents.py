import torch
import numpy as np
from argparse import Namespace
from xuance.common import Optional, Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OnPolicyAgent

class DreamerV2_Agent(OnPolicyAgent):
    pass
    # """The implementation of DreamerV2 agent.
    #
    # Args:
    #     config: ?
    #     envs: ?
    # """
    # def __init__(self,
    #              config: Namespace,
    #              envs: Union[DummyVecEnv, SubprocVecEnv]):
    #     super(DreamerV2_Agent, self).__init__(config, envs)
    #
    #     self.policy = self._build_policy()  # build policy
    #     self.memory = self._build_memory()  # build memory
    #     self.learner = self._build_learner(self.config, self.policy)  # build learner
    #
    # def _build_policy(self) -> Module:
    #     normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
    #     initializer = torch.nn.init.orthogonal_
    #     activation = ActivationFunctions[self.config.activation]
    #     device = self.device
    #
    #     # build representation.
    #     representation = self._build_representation(self.config.representation, self.observation_space, self.config)
    #
    #     # build policy.
    #     if self.config.policy == "Basic_Q_network":
    #         policy = REGISTRY_Policy["Basic_Q_network"](
    #             action_space=self.action_space, representation=representation, hidden_size=self.config.q_hidden_size,
    #             normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
    #             use_distributed_training=self.distributed_training)
    #     else:
    #         raise AttributeError(f"{self.config.agent} does not support the policy named {self.config.policy}.")
    #
    #     return policy


