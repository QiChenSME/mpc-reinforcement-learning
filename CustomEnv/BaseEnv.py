import math
from typing import Optional

import casadi as cs
import gymnasium as gym
import numpy as np

from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from typing import Any, Dict, List, Tuple, Union, Generic, TypeVar
from numpy.typing import NDArray

import ABC

class MPCRLEnv(gym.Env, ABC):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "render_fps": 120,
    }
    nx = None
    nu = None

    def __init__(
            self, *,
            nx:Optional[int]=None,
            nu:Optional[int]=None,
            thresholds_x:Optional[NDArray[Any, dtype[np.float32]]]=None,
            thresholds_u: Optional[NDArray[Any, dtype[np.float32]]] = None,
            bounds:Optional[NDArray[Any, dtype[np.float32]]]=None,
            parameters:Optional[Dict]=None,
            dynamics:Optional[cs.Function]=None,
            jacobian:Optional[cs.Function]=None,
            render_mode: Optional[str] = None,
    ):
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters.copy()

        self.nx = nx
        self.nu = nu
        self.x_sym = cs.SX.sym('x', nx)
        self.u_sym = cs.SX.sym('u', nu)

        if thresholds_x is None:
            self.thresholds_x = np.full(shape=(nx, 2), fill_value=np.inf, dtype=np.float32)
        else:
            self.thresholds_x = thresholds_x.copy()
        if thresholds_u is None:
            self.thresholds_u = np.full(shape=(nu, 2), fill_value=np.inf, dtype=np.float32)
        else:
            self.thresholds_u = thresholds_u.copy()

        self.observation_space = spaces.Box(self.thresholds_x[0], self.thresholds_x[1], dtype=np.float32)
        self.action_space = spaces.Box(self.thresholds_u[0], self.thresholds_u[1], dtype=np.float32)

        self.dynamics = dynamics
        self.jacobian = jacobian

        self.render_mode = render_mode

        if bounds is None:
            self.bounds = thresholds_x.copy()
        else:
            self.bounds = bounds.copy()

        self.screen_width = 1600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.last_x = None
        self.last_u = None
        self.last_r = None

        self.timesteps = 0
        self.steps_beyond_terminated = None

    def _before_step(self):
        pass

    def _after_step(self):
        pass

    def step(self, action:cs.DM):
        self.last_u = action.full().reshape(self.nu, 1)
        self.last_x = self.state.copy()

        self._before_step()

        self.state = np.asarray(self.dynamics(self.state, action).full().flatten()).reshape(self.nx, 1)

        self._after_step()





