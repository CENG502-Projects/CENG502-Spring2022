import logging
import collections

import numpy as np
import torch
from torch import nn

from . import dqn_agent
from ..tools import flag_tools


class DqnReprAgent(dqn_agent.DqnAgent):

    def __init__(
            self, 
            reward_mode='sparse',
            dist_reward_coeff=1.0,
            goal_obs_prepro=None,
            **kwargs):
        assert reward_mode in ['sparse', 'l2', 'mix', 'rawmix']
        self._reward_mode = reward_mode
        self._dist_reward_coeff = dist_reward_coeff
        self._goal_obs_prepro = goal_obs_prepro
        super().__init__(**kwargs)

    def _build_model(self):
        super()._build_model()
        self._repr_fn = self._model.repr_fn

    def _get_goal_obs_batch(self, steps):
        obs_batch = [self._goal_obs_prepro(s.step.time_step.observation)
                for s in steps]
        return np.stack(obs_batch, axis=0)

    def _get_train_batch(self):
        # Add augemented reward
        steps1, steps2 = self._replay_buffer.sample_transitions(
                batch_size=self._batch_size)
        a = self._get_action_batch(steps1)
        s1, s2 = map(self._get_obs_batch, [steps1, steps2])
        sg = self._get_goal_obs_batch(steps2)
        r, dsc = self._get_r_dsc_batch(steps2)
        batch = flag_tools.Flags()
        batch.s1 = self._tensor(s1)
        batch.s2 = self._tensor(s2)
        batch.dsc = self._tensor(dsc)
        batch.a = self._tensor(a)
        batch.r = self._tensor(r)
        batch.sg = self._tensor(sg)
        # overwrite reward
        batch.r = self._get_repr_reward(batch.r, batch.s2, batch.sg)
        return batch

    def _get_repr_reward(self, r, s2, sg):
        if self._reward_mode == 'l2':
            # compute raw distance between s2 and sg
            dist = (s2 - sg).flatten(start_dim=1).norm(dim=-1)
            r = - dist * self._dist_reward_coeff
        elif self._reward_mode == 'rawmix':
            # mix the raw distance with the original reward
            dist = (s2 - sg).flatten(start_dim=1).norm(dim=-1)
            r = - dist * self._dist_reward_coeff * 0.5 + r * 0.5
        elif self._reward_mode == 'mix':
            # mix the learned distance with the original reward
            with torch.no_grad():
                s2_repr = self._repr_fn(s2)
                sg_repr = self._repr_fn(sg) 
            dist = (s2_repr - sg_repr).norm(dim=-1)
            r = - dist * self._dist_reward_coeff * 0.5 + r * 0.5
        # otherwise reward_mode == 'sparse' then keep r
        return r
            

class DqnReprAgentModel(nn.Module):

    def __init__(
            self, 
            q_model_factory, 
            repr_model_factory,
            ):
        super().__init__()
        self.q_fn_learning = q_model_factory()
        self.q_fn_target = q_model_factory()
        self.repr_fn = repr_model_factory()


class DqnReprAgentConfig(dqn_agent.DqnAgentConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        flags.reward_mode = 'sparse'
        flags.dist_reward_coeff = 1.0
        flags.repr_model_cfg = None

    def _repr_model_factory(self):
        """Build model and load from checkpoint."""
        raise NotImplementedError

    def _goal_obs_prepro(self, obs):
        """Get preprocessed goal representation."""
        raise NotImplementedError

    def _model_factory(self):
        return DqnReprAgentModel(
                q_model_factory=self._q_model_factory,
                repr_model_factory=self._repr_model_factory
                )

    def _build_args(self):
        super()._build_args()
        args = self._args
        flags = self._flags
        args.reward_mode = flags.reward_mode
        args.dist_reward_coeff = flags.dist_reward_coeff
        args.goal_obs_prepro = self._goal_obs_prepro












