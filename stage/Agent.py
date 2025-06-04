import sys
from typing import Callable, Generic, Literal, Optional, SupportsFloat, Union
from csnlp import Solution
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias
from mpcrl.agents.common.agent import ActType, ObsType, SymType
from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent, LstdDpgAgent, UpdateStrategy
from mpcrl.util.control import dlqr
from mpcrl.core.exploration import *
from gymnasium import Env


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
        self._hook_callback("terminal_cost_update", "on_env_step", self._terminal_cost_update)

    # def _try_store_experience(
    #         self, cost: SupportsFloat, solQ: Solution[SymType], solV: Solution[SymType]
    # ) -> bool:
    #     """Internal utility that tries to store the gradient and hessian for the current
    #     transition in memory, if both ``V`` and ``Q`` were successful; otherwise, does
    #     not store it. Returns whether it was successful or not."""
    #     success = solQ.success and solV.success
    #     if success:
    #         td_error = float(cost) + self.discount_factor * solV.f - solQ.f
    #         # print("td_error: ", td_error, "cost: ", cost, "solV.f: ", solV.f, "solQ.f: ", solQ.f)
    #         if self.hessian_type == "none":
    #             dQ = self._sensitivity(solQ)
    #             gradient = -td_error * dQ
    #             self.store_experience(gradient)
    #         else:
    #             dQ, ddQ = self._sensitivity(solQ)
    #             gradient = -td_error * dQ
    #             hessian = np.multiply.outer(dQ, dQ) - td_error * ddQ
    #             self.store_experience((gradient, hessian))
    #     else:
    #         td_error = np.nan
    #
    #     if self.td_errors is not None:
    #         self.td_errors.append(td_error)
    #     return success
class DependencyNotInstalled(Exception):
    pass

class LinearLstdQLearningAgent(LstdQLearningAgent):
    def __init__(self, show_rewards=True, *args, **kwargs) -> None:
        self.fig, self.axs, self.line1 = None, None, None
        self.show_rewards = show_rewards
        self.rewards_list = []
        if self.show_rewards:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation
            except ImportError as e:
                raise DependencyNotInstalled(
                    'matplotlib is not installed, run `pip install "matplotlib"`'
                ) from e
            self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 9))
            self.ax.set_yscale('log')
            self.line1, = self.ax.plot([], [], "ob", lw=2)
            self.line2, = self.ax.plot([], [], "-r", lw=2)

        super().__init__(*args, **kwargs)

    def train(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        seed: RngType = None,
        raises: bool = True,
        env_reset_options: Optional[dict[str, Any]] = None,
    ) -> npt.NDArray[np.floating]:
        if self.show_rewards:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation
            except ImportError as e:
                raise DependencyNotInstalled(
                    'matplotlib is not installed, run `pip install "matplotlib"`'
                ) from e
            plt.ion()
            plt.draw()
            plt.pause(0.0001)

        return super().train(env, episodes, seed, raises, env_reset_options)

    def train_one_episode(self, *args) -> float:
        rewards = super().train_one_episode(*args)
        if self.show_rewards:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation
            except ImportError as e:
                raise DependencyNotInstalled(
                    'matplotlib is not installed, run `pip install "matplotlib"`'
                ) from e
            import numpy as np
            self.rewards_list.append(rewards)
            self.ax.set_xlim(0, len(self.rewards_list))
            self.ax.set_ylim(np.min(self.rewards_list)- 0.1, np.max(self.rewards_list) + 0.1)
            self.line1.set_data(range(len(self.rewards_list)), self.rewards_list)
            if len(self.rewards_list)>25:
                coeff = np.polyfit(range(len(self.rewards_list)), self.rewards_list, deg=3)
                reward_smoothed = np.polyval(coeff, np.linspace(0, len(self.rewards_list)-1, len(self.rewards_list)))
                self.line2.set_data(range(len(self.rewards_list)), reward_smoothed)
            # self.ax.plot(self.rewards_list, lw=25
            plt.draw()
            plt.pause(0.001)
        return rewards

    def _show_rewards(self, *args) -> None:
        if self.show_rewards:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            plt.ioff()



    def _terminal_cost_update(self, *args) -> None:
        # print(f"A: {self._fixed_pars['A']}")
        # print(f"B: {self._fixed_pars['B']}")
        self._fixed_pars["Q"] = 0.5 * (self._fixed_pars["Q"] + self._fixed_pars["Q"].T)
        self._fixed_pars["R"] = 0.5 * (self._fixed_pars["R"] + self._fixed_pars["R"].T)
        self._fixed_pars["S"] = dlqr(self._fixed_pars["A"], self._fixed_pars["B"], self._fixed_pars["Q"], self._fixed_pars["R"])[1]
        # print(f"S: {self._fixed_pars['S']}")

    def _dynamic_update(self, *args):
        A, B = self.V.jacobian(self.V.env.state.flatten(), self.V.env.last_action)
        A = A.full().reshape((self.V.env.nx, self.V.env.nx))
        B = B.full().reshape((self.V.env.nx, self.V.env.nu))

        self._fixed_pars["A"] = A
        self._fixed_pars["B"] = B


    def _symmetric_q_r(self):
        self._fixed_pars["Q"] = 0.5 * (self._fixed_pars["Q"] + self._fixed_pars["Q"].T)
        self._fixed_pars["R"] = 0.5 * (self._fixed_pars["R"] + self._fixed_pars["R"].T)
        # print(self._fixed_pars)

    def _establish_callback_hooks(self) -> None:
        super()._establish_callback_hooks()
        self._hook_callback("symmetric_q_r", "on_update", self._symmetric_q_r)
        self._hook_callback("terminal_cost_update", "on_env_step", self._terminal_cost_update)
        self._hook_callback("dynamic_update", "on_env_step", self._dynamic_update)
