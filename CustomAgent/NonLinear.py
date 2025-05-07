import sys
from collections.abc import Collection, Iterable
from typing import Callable, Generic, Literal, Optional, SupportsFloat, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc, NlpSensitivity
from gymnasium import Env

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.update import UpdateStrategy
from mpcrl.core.warmstart import WarmStartStrategy
from mpcrl.optim.gradient_based_optimizer import GradientBasedOptimizer
from mpcrl.agents.common.agent import ActType, ObsType, SymType
from mpcrl.agents.common.rl_learning_agent import LrType, RlLearningAgent

import logging
from typing import Any, Optional

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import Mpc
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit

from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent, LstdDpgAgent, UpdateStrategy
from mpcrl.optim import NetwonMethod, GradientDescent
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes
from mpcrl.core.exploration import *

from CustomEnv.CartpoleProMax import CartPoleV3, CartPoleV4, CartPoleCS


class NonLinearLstdQLearningAgent(LstdQLearningAgent):
    def _terminal_cost_update(self, *args) -> None:
        A, B = self.V.jacobian(self.V.env.state.flatten(), self.V.env.last_action)
        A = A.full().reshape((self.V.env.nx, self.V.env.nx))
        B = B.full().reshape((self.V.env.nx, self.V.env.nu))

        self._fixed_pars["Q"] = 0.5 * (self._fixed_pars["Q"] + self._fixed_pars["Q"].T)
        self._fixed_pars["R"] = 0.5 * (self._fixed_pars["R"] + self._fixed_pars["R"].T)
        self._fixed_pars["S"] = dlqr(A, B, self._fixed_pars["Q"], self._fixed_pars["R"])[1]

    def _symmetric_q_r(self):
        self._fixed_pars["Q"] = 0.5 * (self._fixed_pars["Q"] + self._fixed_pars["Q"].T)
        self._fixed_pars["R"] = 0.5 * (self._fixed_pars["R"] + self._fixed_pars["R"].T)

    def _establish_callback_hooks(self) -> None:
        super()._establish_callback_hooks()
        self._hook_callback("symmetric_q_r", "on_update", self._symmetric_q_r)
        self._hook_callback("terminal_cost_update", "on_timestep_end", self._terminal_cost_update)

    def _try_store_experience(
            self, cost: SupportsFloat, solQ: Solution[SymType], solV: Solution[SymType]
    ) -> bool:
        """Internal utility that tries to store the gradient and hessian for the current
        transition in memory, if both ``V`` and ``Q`` were successful; otherwise, does
        not store it. Returns whether it was successful or not."""
        success = solQ.success and solV.success
        if success:
            td_error = float(cost) + self.discount_factor * solV.f - solQ.f
            # print("td_error: ", td_error, "cost: ", cost, "solV.f: ", solV.f, "solQ.f: ", solQ.f)
            if self.hessian_type == "none":
                dQ = self._sensitivity(solQ)
                gradient = -td_error * dQ
                self.store_experience(gradient)
            else:
                dQ, ddQ = self._sensitivity(solQ)
                gradient = -td_error * dQ
                hessian = np.multiply.outer(dQ, dQ) - td_error * ddQ
                self.store_experience((gradient, hessian))
        else:
            td_error = np.nan

        if self.td_errors is not None:
            self.td_errors.append(td_error)
        return success


class NonLinearLstdDpgAgent(LstdDpgAgent):
    def terminal_cost_update(self) -> None:
        A, B = self.V.jacobian(self.V.env.state.flatten(), self.V.env.last_action)
        A = A.full().reshape((self.V.env.nx, self.V.env.nx))
        B = B.full().reshape((self.V.env.nx, self.V.env.nu))
        self._fixed_pars["S"] = dlqr(A, B, self._fixed_pars["Q"], self._fixed_pars["R"])[1]

    def _establish_callback_hooks(self) -> None:
        super()._establish_callback_hooks()
        self._hook_callback("terminal_cost_update", "on timestep_end", self.terminal_cost_update)