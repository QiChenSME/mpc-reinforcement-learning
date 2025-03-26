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

from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.optim import NetwonMethod
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes
from mpcrl.core.exploration import *

from CustomEnv.CartpoleProMax import CartPoleV3, CartPoleV4, CartPoleCS


class LinearMpc(Mpc[cs.SX]):
    """A simple linear MPC controller."""
    env = CartPoleV4()

    horizon = 10
    discount_factor = 0.9
    M = env.masscart
    m = env.masspole
    g = env.gravity
    l = env.length
    Ts = env.tau

    C = M * m * l**2

    learnable_pars_init = {
        "V0": np.asarray(0.0),
        "x_lb": np.asarray(env.x_bnd[0]).reshape(4, ),
        "x_ub": np.asarray(env.x_bnd[1]).reshape(4, ),
        "b": np.zeros(env.nx),
        "f": np.zeros(env.nx + env.nu),
        "A": np.eye(4) + np.asarray([[0, 1, 0, 0],
                         [0, 0, (- m**3 * l**4 * g * (M+m) + m**4 * g**2 * l**4)/C, 0],
                         [0, 0, 0, 1],
                         [0, 0, (M+m)*m*g*l, 0]]) * Ts / C,
        "B": np.asarray([[0],
                         [m * l**2],
                         [0],
                         [-m * l]]) * Ts / C,
    }

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        w = self.env.w
        nx, nu = self.env.nx, self.env.nu
        x_bnd, a_bnd = self.env.x_bnd, self.env.a_bnd
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # parameters
        V0 = self.parameter("V0")
        x_lb = self.parameter("x_lb", (nx,))
        x_ub = self.parameter("x_ub", (nx,))
        b = self.parameter("b", (nx, 1))
        f = self.parameter("f", (nx + nu, 1))
        A = self.parameter("A", (nx, nx))
        B = self.parameter("B", (nx, nu))

        # variables (state, action, slack)
        x, _ = self.state("x", nx, bound_initial=False)
        u, _ = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s, _, _ = self.variable("s", (nx, N), lb=0)

        # dynamics
        self.set_affine_dynamics(A, B, c=b)

        # other constraints
        self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)

        # objective
        A_init, B_init = self.learnable_pars_init["A"], self.learnable_pars_init["B"]
        S = cs.DM(dlqr(A_init, B_init, 0.5 * np.eye(nx), 0.25 * np.eye(nu))[1])
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            V0
            + cs.bilin(S, x[:, -1])
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w.T @ s)
            )
        )

        # solver
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            "fatrop": {"max_iter": 1000, "print_level": 0},
        }
        self.init_solver(opts, solver="fatrop", type="nlp")

class NonLinearMpc(Mpc[cs.SX]):
    """A simple nonlinear MPC controller."""
    env = CartPoleCS()
    env.reset()

    horizon = 10
    discount_factor = 0.9
    M = env.masscart
    m = env.masspole
    g = env.gravity
    l = env.length
    Ts = env.tau

    C = M * m * l ** 2

    A_init, B_init = env.jacobian(env.state.flatten(), np.zeros(env.nu))
    A_init = A_init.full().reshape((env.nx, env.nx))
    B_init = B_init.full().reshape((env.nx, env.nu))

    learnable_pars_init = {
        "V0": np.asarray(0.0),
        "x_lb": np.asarray(env.x_bnd[0]).reshape(4, ),
        "x_ub": np.asarray(env.x_bnd[1]).reshape(4, ),
        "f": np.zeros(env.nx + env.nu),
        "Q": 0.5 * np.eye(env.nx),
        "R": 0.25 * np.eye(env.nu),
    }
    fixed_pars_init = {
        "S": dlqr(A_init,
                  B_init,
                  learnable_pars_init["Q"],
                  learnable_pars_init["R"])[1]
    }

    def __init__(self, *args, **kwargs) -> None:
        N = self.horizon
        gamma = self.discount_factor
        w = self.env.w
        nx, nu = self.env.nx, self.env.nu
        x_bnd, a_bnd = self.env.x_bnd, self.env.a_bnd
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        self.dynamics = self.env.dynamics
        self.jacobian = self.env.jacobian

        # parameters
        V0 = self.parameter("V0")
        x_lb = self.parameter("x_lb", (nx,))
        x_ub = self.parameter("x_ub", (nx,))
        f = self.parameter("f", (nx + nu, 1))

        Q = self.parameter("Q", (nx, nx))
        R = self.parameter("R", (nu, nu))
        S = self.parameter("S", (nx, nx))

        # variables (state, action, slack)
        x, _ = self.state("x", nx, bound_initial=False)
        u, _ = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s, _, _ = self.variable("s", (nx, N), lb=0)

        # dynamics
        self.set_nonlinear_dynamics(self.dynamics)

        # other constraints
        self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)

        # objective

        # A_init, B_init = self.fixed_pars_init["A"], self.fixed_pars_init["B"]
        S_init = cs.DM(dlqr(self.A_init, self.B_init, 0.5 * np.eye(nx), 0.25 * np.eye(nu))[1])

        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            # V0
            # + cs.bilin(Q, x[:, 0])
            # + cs.bilin(R, u[:, 0])
            + cs.bilin(S, x[:, -1])
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers * (cs.sum1(Q @ x[:, :-1] * x[:, :-1]) + 0.5 * cs.sum1(R @ u * u) + w.T @ s)
            )
        )

        # solver
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            "fatrop": {"max_iter": 1000, "print_level": 0},
        }
        self.init_solver(opts, solver="fatrop", type="nlp")

    def minimize_update(self):
        N = self.horizon
        gamma = self.discount_factor
        w = self.env.w
        nx, nu = self.env.nx, self.env.nu
        x_bnd, a_bnd = self.env.x_bnd, self.env.a_bnd

        # parameters
        V0 = self.parameter("V0")
        f = self.parameter("f", (nx + nu, 1))

        # variables (state, action, slack)
        x, _ = self.state("x", nx, bound_initial=False)
        u, _ = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s, _, _ = self.variable("s", (nx, N), lb=0)

        A, B = self.jacobian(self.env.state.flatten(), np.zeros(self.env.nu))
        A = A.full().reshape((nx, nx))
        B = B.full().reshape((nx, nu))
        S = cs.DM(dlqr(A, B, 0.5 * np.eye(nx), 0.25 * np.eye(nu))[1])
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            # V0
            + cs.bilin(S, x[:, -1])
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u ** 2) + w.T @ s)
            )
        )

    def para_update(self):
        pass


class NonLinearLstdQLearningAgent(LstdQLearningAgent):
    def terminal_cost_update(self) -> None:
        A, B = self.V.jacobian(self.V.env.state.flatten(), self.V.env.last_action)
        A = A.full().reshape((self.V.env.nx, self.V.env.nx))
        B = B.full().reshape((self.V.env.nx, self.V.env.nu))
        self._fixed_pars["S"] = dlqr(A, B, self._fixed_pars["Q"], self._fixed_pars["R"])[1]

    def _establish_callback_hooks(self) -> None:
        super()._establish_callback_hooks()
        self._hook_callback("terminal_cost_update", "on timestep_end", self.terminal_cost_update)


if __name__ == "__main__":
    import os

    # instantiate the env and wrap it
    render_mode = None
    render_mode = "human"
    mpc_type = "Linear"
    mpc_type = "NonLinear"
    env = MonitorEpisodes(TimeLimit(CartPoleCS(render_mode=render_mode), max_episode_steps=1000))
    # now build the MPC and the dict of learnable parameters
    if mpc_type == "NonLinear":
        mpc = NonLinearMpc()
    else:
        mpc = LinearMpc()
    learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
            for name, val in mpc.learnable_pars_init.items()
        )
    )

    # build and wrap appropriately the agent
    # noinspection PyTypeChecker
    if mpc_type == "NonLinear":
        # noinspection PyTypeChecker
        agent = Log(
            RecordUpdates(
                NonLinearLstdQLearningAgent(
                    mpc=mpc,
                    learnable_parameters=learnable_pars,
                    fixed_parameters=mpc.fixed_pars_init,
                    discount_factor=mpc.discount_factor,
                    update_strategy=20,
                    optimizer=NetwonMethod(learning_rate=0),
                    hessian_type="approx",
                    record_td_errors=True,
                    remove_bounds_on_initial_action=True,
                    use_last_action_on_fail=True,
                    # exploration=EpsilonGreedyExploration(0.01, 1, hook="on_timestep_end"),
                )
            ),
            level=logging.DEBUG,
            log_frequencies={"on_timestep_end": 1000},
        )
    else:
        # noinspection PyTypeChecker
        agent = Log(
            RecordUpdates(
                LstdQLearningAgent(
                    mpc=mpc,
                    learnable_parameters=learnable_pars,
                    discount_factor=mpc.discount_factor,
                    update_strategy=200,
                    optimizer=NetwonMethod(learning_rate=3e-2),
                    hessian_type="approx",
                    record_td_errors=True,
                    remove_bounds_on_initial_action=True,
                    use_last_action_on_fail=True,

                )
            ),
            level=logging.DEBUG,
            log_frequencies={"on_timestep_end": 1000},
        )

    # launch the training simulation
    try:
        agent.train(env=env, episodes=500000, seed=69, raises=False)
    except SystemError:
        pass
    finally:
        print("\nUser Interrupted, training aborted.")

        import matplotlib.pyplot as plt
        import os
        import sys

        img_path = "images"
        os.makedirs(img_path, exist_ok=True)

        X = env.get_wrapper_attr("observations")[-1].squeeze().T
        U = env.get_wrapper_attr("actions")[-1].squeeze()
        R = env.get_wrapper_attr("rewards")[-1]
        T_R = env.get_wrapper_attr("rewards")
        RWD = list(map(sum,T_R))
        STP = list(map(len,T_R))

        _, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
        axs[0].plot(X[0])
        axs[1].plot(X[1])
        axs[2].plot(X[2])
        axs[3].plot(X[3])
        axs[4].plot(U)
        for i in range(2):
            # axs[0].axhline(env.get_wrapper_attr("x_bnd")[i][0], color="r")
            axs[2].axhline(env.get_wrapper_attr("x_bnd")[i][2], color="r")
            # axs[4].axhline(env.get_wrapper_attr("a_bnd")[i], color="r")
        axs[0].set_ylabel("$X$")
        axs[1].set_ylabel("$X'$")
        axs[2].set_ylabel(r"$\theta$")
        axs[3].set_ylabel(r"$\theta'$")
        axs[4].set_ylabel("$a$")

        img_name = "state_last_episode.svg"
        path = os.path.join(img_path, img_name)
        plt.savefig(path, format="svg")

        _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        axs[0].plot(agent.td_errors[-len(R):-1], "o", markersize=1)
        axs[1].semilogy(R, "o", markersize=1)
        axs[0].set_ylabel(r"$\tau$")
        axs[1].set_ylabel("$L$")

        img_name = "td_error_and_loss_last_episode.svg"
        path = os.path.join(img_path, img_name)
        plt.savefig(path, format="svg")

        _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        axs[0].semilogy(RWD, "ro-", markersize=4)
        axs[0].set_ylabel("$L$")
        axs[1].plot(STP, "bo-", markersize=4)
        axs[1].set_ylabel("steps")

        img_name = "episodes_loss_and_steps.svg"
        path = os.path.join(img_path, img_name)
        plt.savefig(path, format="svg")

        if mpc_type == "NonLinear":
            _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
            axs[0].plot(
                np.stack(
                    [np.asarray(agent.updates_history[n])[:, 0] for n in ("x_lb", "x_ub")], -1
                ),
            )
            axs[1].plot(np.asarray(agent.updates_history["f"]))
            axs[2].plot(np.asarray(agent.updates_history["V0"]))
            axs[0].set_ylabel("$x_1$")
            axs[1].set_ylabel("$f$")
            axs[2].set_ylabel("$V_0$")
            _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
            axs[0].plot(np.asarray(agent.updates_history["Q"]).reshape(-1, mpc.env.nx**2))
            axs[1].plot(np.asarray(agent.updates_history["R"]).reshape(-1,mpc.env.nu**2))
            axs[0].set_ylabel("$Q$")
            axs[1].set_ylabel("$R$")
        else:
            _, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
            axs[0, 0].plot(np.asarray(agent.updates_history["b"]))
            axs[0, 1].plot(
                np.stack(
                    [np.asarray(agent.updates_history[n])[:, 0] for n in ("x_lb", "x_ub")], -1
                ),
            )
            axs[1, 0].plot(np.asarray(agent.updates_history["f"]))
            axs[1, 1].plot(np.asarray(agent.updates_history["V0"]))
            axs[2, 0].plot(np.asarray(agent.updates_history["A"]).reshape(-1,16))
            axs[2, 1].plot(np.asarray(agent.updates_history["B"]).squeeze())
            axs[0, 0].set_ylabel("$b$")
            axs[0, 1].set_ylabel("$x_1$")
            axs[1, 0].set_ylabel("$f$")
            axs[1, 1].set_ylabel("$V_0$")
            axs[2, 0].set_ylabel("$A$")
            axs[2, 1].set_ylabel("$B$")

        img_name = "para.svg"
        path = os.path.join(img_path, img_name)
        plt.savefig(path, format="svg")

        plt.show()

        sys.exit(0)
