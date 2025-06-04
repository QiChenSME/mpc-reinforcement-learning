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
                 render_mode: Optional[str] = None,
                 debug_mode: bool = False
                 ):
        self.model = model

        self._dynamics = self.model.integrator
        self._jacobian = self.model.jacobian
        self._sym_x = self.model.sym_x
        self._sym_u = self.model.sym_u

        self._nx = model.nx
        self._nu = model.nu

        if x_bnd is None:
            self.x_bnd = (-np.ones((self._nx, 1))*0.5, np.ones((self._nx, 1))*0.5)
        else:
            self.x_bnd = x_bnd
        if u_bnd is None:
            self.u_bnd = (-np.ones((self._nu, 1))*1000, np.ones((self._nu, 1))*1000)
        else:
            self.u_bnd = u_bnd
        if w_x is None:
            self.w_x = np.ones((self._nx, 1)) * 100
        else:
            self.w_x = np.asarray(w_x)
        if w_u is None:
            self.w_u = np.ones((self._nu, 1)) * 0
        else:
            self.w_u = np.asarray(w_u)

        self.action_space = spaces.Box(np.asarray(self.u_bnd[0],dtype=np.float32).flatten(),
                                       np.asarray(self.u_bnd[1],dtype=np.float32).flatten(), dtype=np.float32)
        self.observation_space = spaces.Box(np.asarray(self.x_bnd[0],dtype=np.float32).flatten(),
                                            np.asarray(self.x_bnd[1],dtype=np.float32).flatten(), dtype=np.float32)

        self.state = None
        self.last_action = None

        self.render_mode = render_mode
        self.debug_mode = debug_mode

        # 清空越界判定
        self.steps_beyond_terminated = None

    @property
    def nx(self):
        return self._nx

    @property
    def nu(self):
        return self._nu

    @property
    def dynamics(self):
        x_f = self._dynamics(x0=self._sym_x, p=self._sym_u)['xf']
        return cs.Function('dynamics', [self._sym_x, self._sym_u], [x_f])

    @property
    def jacobian(self):
        return self._jacobian

    def step(self,
             action:cs.DM | np.ndarray,
             ref = None
             ):
        assert self.state is not None, "Call reset before using step method."

        if ref is None:
            ref = np.ones((self._nx, 1))
        else:
            ref = np.asarray(ref, dtype=np.float32).reshape(self._nx, 1)

        if type(action) is cs.DM:
            action = action.full().flatten()
        elif type(action) is np.ndarray:
            action = action.flatten()
        action = cs.DM(np.clip(action, self.u_bnd[0].flatten(), self.u_bnd[1].flatten()))
        self.last_action = action

        self.state = self._dynamics(x0=self.state.flatten(), p=action)['xf'].full().flatten()
        # self.state = self.dynamics(self.state.flatten(), action).full().flatten()
        self.state = np.clip(self.state, self.x_bnd[0].flatten()*2, self.x_bnd[1].flatten()*2).reshape(self._nx, 1)

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

        if self.render_mode == "human":
            self.render()
        if self.debug_mode:
            print(f"State: {self.state.flatten()}")
            print(f"Action: {action.full().flatten()}")
            print(f"Reward: {reward}")

        return np.array(self.state, dtype=np.float32), reward, False, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # self.state = self.np_random.uniform(low=self.x_bnd[0].flatten(),
        #                                     high=self.x_bnd[1].flatten(),
        #                                     size=(self._nx,1)).astype(np.float32).reshape(self._nx, 1)
        self.state = np.zeros((self._nx, 1)).reshape(self._nx, 1)
        # self.state = np.asarray([0.0005, 0.0005, 0.0005, 0.0001, 0.0001, 0.0001, 0, 0, 0, 0, 0, 0]).reshape(self._nx, 1)
        self.last_action = 0

        self.steps_beyond_terminated = None

        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        raise NotImplementedError("Render method not implemented.")


class StageEnv(CasadiEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._first_render = True
        self.data_len = 100
        self.state_history = np.zeros((0, 6))
        self.ref_history = np.zeros((0, 6))
        self.update_rate = 10
        self.frame_count = 0

        self.last_ref = None
        self._first_render = True
        self.fig, self.axs = None, None
        self.line1, self.line2, self.line3, self.line4, self.line5, self.line6 = None, None, None, None, None, None
        self.line7, self.line8, self.line9, self.line10, self.line11, self.line12 = None, None, None, None, None, None

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        ret = super().reset(seed=seed)
        # self.state = np.asarray([0.0005, 0.0005, 0.0005, 0.0001, 0.0001, 0.0001, 0, 0, 0, 0, 0, 0]).reshape(self._nx, 1)
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.005, 0.005  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low,
                                            high=high,
                                            size=(self._nx,1)).astype(np.float32).reshape(self._nx, 1)
        self.last_ref = np.zeros((self._nx, 1)).reshape(self._nx, 1)
        if self.render_mode == "human":
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation
            except ImportError as e:
                raise DependencyNotInstalled(
                    'matplotlib is not installed, run `pip install "matplotlib"`'
                ) from e
            if not self._first_render:
                plt.ioff()
                plt.close(self.fig)
                self._first_render = True
                self.state_history = np.zeros((0, 6))
                self.ref_history = np.zeros((0, 6))

            # 创建图形和轴
            self.fig, self.axs = plt.subplots(6, 1, figsize=(10, 12))
            # 初始化图形
            self.line1, = self.axs[0].plot([], [], lw=2)
            self.line2, = self.axs[1].plot([], [], lw=2)
            self.line3, = self.axs[2].plot([], [], lw=2)
            self.line4, = self.axs[3].plot([], [], lw=2)
            self.line5, = self.axs[4].plot([], [], lw=2)
            self.line6, = self.axs[5].plot([], [], lw=2)
            self.line7, = self.axs[0].plot([], [], lw=2)
            self.line8, = self.axs[1].plot([], [], lw=2)
            self.line9, = self.axs[2].plot([], [], lw=2)
            self.line10, = self.axs[3].plot([], [], lw=2)
            self.line11, = self.axs[4].plot([], [], lw=2)
            self.line12, = self.axs[5].plot([], [], lw=2)
            self.axs[0].set_xlim(0, self.data_len)  # x轴显示最新的50组数据
            self.axs[1].set_xlim(0, self.data_len)  # x轴显示最新的50组数据
            self.axs[2].set_xlim(0, self.data_len)  # x轴显示最新的50组数据
            self.axs[3].set_xlim(0, self.data_len)  # x轴显示最新的50组数据
            self.axs[4].set_xlim(0, self.data_len)  # x轴显示最新的50组数据
            self.axs[5].set_xlim(0, self.data_len)  # x轴显示最新的50组数据
            self.axs[0].set_ylabel("X")
            self.axs[1].set_ylabel("Y")
            self.axs[2].set_ylabel("Z")
            self.axs[3].set_ylabel(r"\theta x")
            self.axs[4].set_ylabel(r"\theta y")
            self.axs[5].set_ylabel(r"\theta z")

        return np.array(self.state, dtype=np.float32), {}

    def step(self,
             action:cs.DM | np.ndarray,
             ref = None
             ):
        if ref is None:
            ref = np.ones((self._nx, 1))
        else:
            ref = np.asarray(ref, dtype=np.float32).reshape(self._nx, 1)
        self.last_ref = ref
        return super().step(action, ref)

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

        if self._first_render:
            # 开启交互模式
            plt.ion()
            plt.draw()
            plt.pause(0.0001)  # 这里可以调整刷新频率
            self._first_render = False

        self.state_history = np.append(self.state_history, self.state.flatten()[:6].reshape(1, -1), axis=0)
        self.ref_history = np.append(self.ref_history, self.last_ref.flatten()[:6].reshape(1, -1), axis=0)
        self.frame_count += 1

        if self.debug_mode:
            print(f"Frame: {self.frame_count}")

        if self.frame_count > self.update_rate-1:
            if len(self.state_history) > self.data_len:
                self.state_history = self.state_history[-self.data_len:]
                self.ref_history = self.ref_history[-self.data_len:]
            state_data = np.array(self.state_history).T
            ref_data = np.array(self.ref_history).T

            data1 = np.concatenate((state_data[0], ref_data[0]), axis=0)
            data2 = np.concatenate((state_data[1], ref_data[1]), axis=0)
            data3 = np.concatenate((state_data[2], ref_data[2]), axis=0)
            data4 = np.concatenate((state_data[3], ref_data[3]), axis=0)
            data5 = np.concatenate((state_data[4], ref_data[4]), axis=0)
            data6 = np.concatenate((state_data[5], ref_data[5]), axis=0)

            self.axs[0].set_ylim(np.min(data1) - 1e-9, np.max(data1) + 1e-9)  # y轴的范围可以根据数据的范围进行调整
            self.axs[1].set_ylim(np.min(data2) - 1e-9, np.max(data2) + 1e-9)  # y轴的范围可以根据数据的范围进行调整
            self.axs[2].set_ylim(np.min(data3) - 1e-9, np.max(data3) + 1e-9)  # y轴的范围可以根据数据的范围进行调整
            self.axs[3].set_ylim(np.min(data4) - 1e-9, np.max(data4) + 1e-9)  # y轴的范围可以根据数据的范围进行调整
            self.axs[4].set_ylim(np.min(data5) - 1e-9, np.max(data5) + 1e-9)  # y轴的范围可以根据数据的范围进行调整
            self.axs[5].set_ylim(np.min(data6) - 1e-9, np.max(data6) + 1e-9)  # y轴的范围可以根据数据的范围进行调整

            # 更新线条数据
            self.line1.set_data(range(len(state_data[0])), state_data[0])
            self.line2.set_data(range(len(state_data[1])), state_data[1])
            self.line3.set_data(range(len(state_data[2])), state_data[2])
            self.line4.set_data(range(len(state_data[3])), state_data[3])
            self.line5.set_data(range(len(state_data[4])), state_data[4])
            self.line6.set_data(range(len(state_data[5])), state_data[5])
            self.line7.set_data(range(len(ref_data[0])), ref_data[0])
            self.line8.set_data(range(len(ref_data[1])), ref_data[1])
            self.line9.set_data(range(len(ref_data[2])), ref_data[2])
            self.line10.set_data(range(len(ref_data[3])), ref_data[3])
            self.line11.set_data(range(len(ref_data[4])), ref_data[4])
            self.line12.set_data(range(len(ref_data[5])), ref_data[5])

            # 刷新图形
            plt.draw()
            plt.pause(0.001)  # 这里可以调整刷新频率

            self.frame_count = 0

    def close(self):
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close(self.fig)


class DisturbanceStageEnv(StageEnv):
    def __init__(self, disturbance: Union = None, noise: Union = None, time_delay: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if disturbance is None:
            self.disturbance = np.zeros((self._nu, 1))
        else:
            self.disturbance = np.asarray(disturbance).reshape(self._nu, 1)
        if noise is None:
            self.noise = np.zeros((self._nx, 1))
        else:
            self.noise = np.asarray(noise).reshape(self._nx, 1)
        if time_delay is None:
            self.time_delay = 0
        else:
            self.time_delay = time_delay
        self.action_queue = []

    def step(self, action, *args, **kwargs):
        action += self.np_random.uniform(low=-self.disturbance, high=self.disturbance)
        self.action_queue.append(action)
        if self.debug_mode:
            print("Length of action queue", len(self.action_queue))
        # print(len(self.action_queue))
        if len(self.action_queue) > self.time_delay:
            action = cs.DM(self.action_queue.pop(0))
        else:
            action = cs.DM(np.zeros((self._nu, 1)))
        state, reward, terminated, truncated, info = super().step(action, *args, **kwargs)
        state += self.np_random.uniform(low=-self.noise, high=self.noise)
        return np.array(state, dtype=np.float32), reward, terminated, truncated, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        ret = super().reset(seed=seed, options=options)
        self.action_queue.clear()
        return ret


class CasadiWrapper(gym.Wrapper):
    def __init__(self, env: CasadiEnv):
        super().__init__(env)

from gymnasium.core import ActType, ObsType
from copy import deepcopy
from gymnasium.envs.registration import EnvSpec
from typing import Any, SupportsFloat
class Tracer(gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs):
    def __init__(self,
                 env: CasadiEnv,
                 max_episode_steps: int,
                 mode: str = "static",
                 init_values = None,
                 trans_list = None,
    ):
        def vectorized_generate(length, init_value, input_dict):
            sorted_keys = sorted(input_dict.keys())
            sorted_values = [input_dict[k] for k in sorted_keys]

            # 计算每个键对应的重复次数
            starts = sorted_keys
            ends = sorted_keys[1:] + [length]
            repeats = [end - start for start, end in zip(starts, ends)]

            # 直接生成最终数组
            return np.vstack([np.repeat([init_value], starts[0], axis=0)] +
                             [np.repeat([val], rep, axis=0) for val, rep in zip(sorted_values, repeats)])

        assert (
                isinstance(max_episode_steps, int) and max_episode_steps > 0
        ), f"Expect the `max_episode_steps` to be positive, actually: {max_episode_steps}"
        gym.utils.RecordConstructorArgs.__init__(
            self, max_episode_steps=max_episode_steps
        )
        super().__init__(env)
        if init_values is None:
            init_values = np.zeros(env.nx)
        else:
            init_values = np.array(init_values)
        if trans_list is None:
            self.value_list = np.vstack([init_values]*max_episode_steps)
        else:
            self.value_list = vectorized_generate(max_episode_steps, init_values, trans_list)
        self.mode = mode
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """
        if self.mode == "static":
            observation, reward, terminated, truncated, info = self.env.step(action, ref=self.value_list[0])
        elif self.mode == "random":
            observation, reward, terminated, truncated, info = self.env.step(action, ref=np.random.rand(self.env.nx))
        else:
            observation, reward, terminated, truncated, info = self.env.step(action, ref=self.value_list[self._elapsed_steps])
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            seed: Seed for the environment
            options: Options for the environment

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return super().reset(seed=seed, options=options)

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to include the `max_episode_steps=self._max_episode_steps`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            try:
                env_spec = deepcopy(env_spec)
                env_spec.max_episode_steps = self._max_episode_steps
            except Exception as e:
                gym.logger.warn(
                    f"An exception occurred ({e}) while copying the environment spec={env_spec}"
                )
                return None

        self._cached_spec = env_spec
        return env_spec


if __name__ == '__main__':
    model = WaferStage()
    env = StageEnv(model)
    print(env.dynamics)
    print(env._dynamics)
