import torch
import numpy as np
from argparse import Namespace
from xuance.common import Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OnPolicyAgent

class DreamerV2_Agent(OnPolicyAgent):
    def __init__(self,
                 config: Namespace,
                 ):
        super(DreamerV2_Agent, self).__init__(config, )
        self.memory = self._build_memory()
        self.policy = self._build_policy()
        self.learner = self._build_learner(self.config, self.policy)    # how to learn world model?

    def _build_policy(self) -> Module:

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.

        return 1

    def get_terminated_values(self, observations_next: np.ndarray, rewards: np.ndarray = None):
        """Returns values for terminated states.

        Parameters:
            observations_next (np.ndarray): The terminal observations.
            rewards (np.ndarray): The rewards for terminated states.

        Returns:
            values_next: The values for terminal states.
        """
        values_next = self._process_reward(rewards)
        return values_next

