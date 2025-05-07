from typing import Optional, Union

from gymnasium.core import RenderFrame

from Dynamics import WaferStage
import time
import numpy as np
import gymnasium as gym
import casadi as cs

from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

class CasadiEnv(gym.Env):
    def __init__(self,
                 model:WaferStage,
                 x_bnd:Union = None,
                 u_bnd:Union = None,
                 w_x = None,
                 w_u = None,
                 ):
        self.model = model

        self._dynamics = self.model.integrator
        self._jacobian = self.model.jacobian
        self._sym_x = self.model.sym_x
        self._sym_u = self.model.sym_u

        self._nx = model.nx
        self._nu = model.nu

        self.x_bnd = x_bnd
        self.u_bnd = u_bnd
        self.w_x = w_x
        self.w_u = w_u

        self.action_space = spaces.Box(np.asarray(u_bnd[0],dtype=np.float32).flatten(),
                                       np.asarray(u_bnd[1],dtype=np.float32).flatten(), dtype=np.float32)
        self.observation_space = spaces.Box(np.asarray(x_bnd[0],dtype=np.float32).flatten(),
                                            np.asarray(x_bnd[1],dtype=np.float32).flatten(), dtype=np.float32)

        self.state = None

        # 清空越界判定
        self.steps_beyond_terminated = None

    def step(self,
             action:cs.DM,
             ref = None
             ):
        assert self.state is not None, "Call reset before using step method."

        if ref is None:
            ref = np.zeros((self._nx, 1))
        else:
            ref = np.asarray(ref, dtype=np.float32).reshape(self._nx, 1)

        action = action.full().flatten()
        action = cs.DM(np.clip(action, self.u_bnd[0].flatten(), self.u_bnd[1].flatten()))

        self.state = self._dynamics(x0=self.state.flatten(), p=action)['xf'].full().flatten()
        self.state = np.clip(self.state, self.x_bnd[0].flatten(), self.x_bnd[1].flatten()).reshape(self._nx, 1)

        lb, ub = self.x_bnd[0].reshape(self._nx, 1), self.x_bnd[1].reshape(self._nx, 1)

        reward = float(
            0.5
            * (
                    self.w_x.T @ ((self.state - ref) ** 2)
                    + self.w_u.T @ (action ** 2)
                    + self.w_x.T @ np.maximum(0, lb - self.state)
                    + self.w_x.T @ np.maximum(0, self.state - ub)
            )
        )

        return np.array(self.state, dtype=np.float32), reward, False, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=self.x_bnd[0].flatten(),
                                            high=self.x_bnd[1].flatten(),
                                            size=(self._nx,1)).astype(np.float32).reshape(self._nx, 1)
        self.steps_beyond_terminated = None

        return np.array(self.state, dtype=np.float32), {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except ImportError as e:
            raise DependencyNotInstalled(
                'matplotlib is not installed, run `pip install "matplotlib"`'
            ) from e


        # 创建图形和轴
        fig, ax = plt.subplots()
        # 初始化图形
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(0, 50)  # x轴显示最新的50组数据
        ax.set_ylim(np.min(data) - 1, np.max(data) + 1)  # y轴的范围可以根据数据的范围进行调整
