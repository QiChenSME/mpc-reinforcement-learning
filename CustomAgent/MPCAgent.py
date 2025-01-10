import sys
from collections.abc import Collection, Iterable
from typing import Any, Callable, Generic, Literal, Optional, SupportsFloat, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc, NlpSensitivity
from gymnasium import Env
from gymnasium.spaces import Box

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
from mpcrl.agents.common.agent import ActType, ObsType, SymType, Agent
from mpcrl.util.seeding import RngType, mk_seed

class MPCAgent(
    Agent[SymType], Generic[SymType]
):
    def __init__(
        self,
        mpc: Mpc[SymType],
        fixed_parameters: Union[
            None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        warmstart: Union[
            Literal["last", "last-successful"], WarmStartStrategy
        ] = "last-successful",
        use_last_action_on_fail: bool = False,
        remove_bounds_on_initial_action: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            mpc=mpc,
            fixed_parameters=fixed_parameters,
            exploration=exploration,
            warmstart=warmstart,
            use_last_action_on_fail=use_last_action_on_fail,
            remove_bounds_on_initial_action=remove_bounds_on_initial_action,
            name=name,
        )

    def run(
        self,
        env: Env[ObsType, ActType],
        seed: RngType = None,
        raises: bool = True,
        env_reset_options: Optional[dict[str, Any]] = None,
    ) -> npt.NDArray[np.floating]:
        if hasattr(env, "action_space"):
            assert isinstance(env.action_space, Box), "Env action space must be a Box,"
        rng = np.random.default_rng(seed)
        self.reset(rng)
        returns = np.zeros(1, float)

        # self.on_training_start(env)
        state, _ = env.reset(seed=mk_seed(rng), options=env_reset_options)
        # self.on_episode_start(env, episode, state)

        truncated = terminated = False
        timestep = 0
        rewards = 0.0
        state = state
        action_space = getattr(env, "action_space", None)

        action, solV = self.state_value(state, False, action_space=action_space)
        if not solV.success:
            self.on_mpc_failure(episode, None, solV.status, raises)

        while not (truncated or terminated):
            # compute Q(s,a)
            solQ = self.action_value(state, action)

            # step the system with action computed at the previous iteration
            new_state, cost, truncated, terminated, _ = env.step(action)
            self.on_env_step(env, episode, timestep)

            # compute V(s+) and store transition
            new_action, solV = self.state_value(
                new_state, False, action_space=action_space
            )
            if not (solQ.success and solV.success):
                self.on_mpc_failure(
                    episode, timestep, f"{solQ.status} (Q); {solV.status} (V)", raises
                )

            # increase counters
            state = new_state
            action = new_action
            rewards += float(cost)
            timestep += 1
            self.on_timestep_end(env, episode, timestep)

        # self.on_episode_end(env, episode, r)
        returns[0] = rewards

        # self.on_training_end(env, returns)
        return returns
