import math
from typing import Optional, Tuple, Union

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

class MassBlockEnv(gym.Env[npt.NDArray[np.floating], float]):

    nx = 2
    nu = 1
    x_threshold = 10
    x_dot_threshold = 10
    action_threshold = 10
    x_bnd = (np.asarray([[-x_threshold], [-x_dot_threshold]]), np.asarray([[x_threshold], [x_dot_threshold]]))
    a_bnd = (-action_threshold, action_threshold)

    w = np.asarray([[1e2], [1e2]])
    high = np.array(
        [
            x_threshold * 2,
            x_dot_threshold * 2,
        ],
        dtype=np.float32,
    )

    action_space = spaces.Box(-action_threshold, action_threshold, dtype=np.float32)
    observation_space = spaces.Box(-high, high, dtype=np.float32)

    def __init__(
        self,
            m: float = 1.0,
            c: float = 0.1,
            k: float = 0,
            render_mode: Optional[str] = None
    ):
        self.m = m
        self.c = c
        self.k = k
        self.Ts = 0.01

        self.A = np.asarray([[1, self.Ts], [-(self.k * self.Ts)/self.m, 1 - (self.c * self.Ts)/self.m]])
        self.B = np.asarray([[0], [self.Ts/self.m]])
        self.C = np.asarray([1, 0])
        self.D = np.asarray([0])

        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None

    def step(self, action:cs.DM):
        force = np.float32(action)
        # assert self.action_space.contains(
        #     force
        # ), f"{force!r} ({type(force)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        x, x_dot = self.state

        x_new = np.asarray(self.A @ np.asarray(([x, x_dot]), dtype=np.float64).reshape(self.nx,1) + self.B * force, dtype=np.float64)

        self.state = np.array(x_new, dtype=np.float64)

        terminated = bool(
            (x < -self.x_threshold) |
            (x > self.x_threshold) |
            (x_dot < -self.x_dot_threshold) |
            (x_dot > self.x_dot_threshold)
        )
        # 当没有终止时给予奖励（或不给予负奖励）
        if not terminated:
            lb, ub = self.x_bnd
            reward = float(
            0.5
            * (
                np.square(x_new).sum()
                + 0.5 * action**2
                + self.w.T @ np.maximum(0, lb - x_new)
                + self.w.T @ np.maximum(0, x_new - ub)
            )
        )
            pass
        # 若达到终止且终止判定未更新，更新终止判定
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0

        if self.render_mode == "human":
            self.render()

        # return np.array(x_new, dtype=np.float64), reward, terminated, False, {}
        return np.array(x_new, dtype=np.float64), reward, False, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -8, 8  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(2, 1))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}


