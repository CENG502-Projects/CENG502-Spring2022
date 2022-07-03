import logging
import collections

import numpy as np
import torch
from torch import nn
from torch import optim

from . import agent
from ..tools import flag_tools


class DqnAgent(agent.Agent):

    def _build_model(self):
        cfg = self._model_cfg
        self._model = cfg.model_factory()
        self._model.to(device=self._device)
        self._q_fn_learning = self._model.q_fn_learning
        self._q_fn_target = self._model.q_fn_target
        self._q_fn_target.load_state_dict(
                self._q_fn_learning.state_dict())
        self._vars_learning = self._q_fn_learning.state_dict()
        self._vars_target = self._q_fn_target.state_dict()

    def _build_optimizer(self):
        cfg = self._optimizer_cfg
        self._optimizer = cfg.optimizer_factory(
                self._q_fn_learning.parameters())
            
    def _build_loss(self, batch):
        # modules and tensors
        s1 = batch.s1
        s2 = batch.s2
        a = batch.a
        r = batch.r
        dsc = batch.dsc
        batch_size = a.shape[0]
        ######################
        # networks
        q_vals_learning = self._q_fn_learning(s1)
        q_val_learning = q_vals_learning[torch.arange(batch_size), a]
        q_vals_target = self._q_fn_target(s2)
        val_target = q_vals_target.max(-1)[0]
        q_val_target = (r + dsc * val_target).detach()
        loss = (q_val_learning - q_val_target).pow(2).mean()
        # build print info
        info = self._train_info
        info['q_loss'] = loss.item()
        info['mean_q'] = q_val_target.mean().item()
        info['min_q'] = q_val_target.min().item()
        info['max_q'] = q_val_target.max().item()
        info['mean_r'] = r.mean().item()
        info['mean_dsc'] = dsc.mean().item()
        # 
        # for i, w in enumerate(self._q_fn_learning.parameters()):
        #     print(i, w.sum().item())
        return loss

    def _policy_fn(self, state):
        time_step, _ = state
        s = np.expand_dims(self._obs_prepro(time_step.observation), 0)
        s = self._tensor(s)
        with torch.no_grad():
            q_vals = self._q_fn_learning(s).cpu().numpy()
        return q_vals[0]

    def _train_policy_fn(self, state):
        # epsilon greedy
        q_vals = self._policy_fn(state)
        eps = self._actor_cfg.epsilon_greedy
        if np.random.uniform() <= eps:
            a = self._action_spec.sample()
        else:
            a = np.argmax(q_vals)
        return a, None

    def _test_policy_fn(self, state):
        q_vals = self._policy_fn(state)
        return np.argmax(q_vals), None

    def save_ckpt(self, filepath):
        torch.save(self._model.state_dict(), filepath)


class DqnAgentModel(nn.Module):

    def __init__(self, q_model_factory):
        super().__init__()
        self.q_fn_learning = q_model_factory()
        self.q_fn_target = q_model_factory()


class DqnAgentConfig(agent.AgentConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        flags.actor_cfg = flag_tools.Flags(epsilon_greedy=0.2)
        flags.opt_args = flag_tools.Flags(name='Adam', lr=0.001)

    def _q_model_factory(self):
        raise NotImplementedError

    def _model_factory(self):
        return DqnAgentModel(self._q_model_factory)

    def _optimizer_factory(self, parameters):
        opt = getattr(optim, self._flags.opt_args.name)
        opt_fn = opt(parameters, lr=self._flags.opt_args.lr)
        return opt_fn









