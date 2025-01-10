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

from CustomEnv.CartpoleProMax import CartPoleV3, CartPoleV4
from CustomAgent.MPCAgent import MPCAgent


class LinearMpc(Mpc[cs.SX]):
    """A simple linear MPC controller."""
    env = CartPoleV3()

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
            "fatrop": {"max_iter": 500, "print_level": 0},
        }
        self.init_solver(opts, solver="fatrop", type="nlp")


if __name__ == "__main__":
    # instantiate the env and wrap it
    render_mode = "human"
    env = MonitorEpisodes(TimeLimit(CartPoleV4(render_mode=render_mode), max_episode_steps=5_00))
    # now build the MPC and the dict of learnable parameters
    mpc = LinearMpc()
    learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
            for name, val in mpc.learnable_pars_init.items()
        )
    )

    # build and wrap appropriately the agent
    # noinspection PyTypeChecker
    agent = MPCAgent(
                mpc=mpc,
                remove_bounds_on_initial_action=True,
                use_last_action_on_fail=True,
            )

    # launch the training simulation
    agent.run(env=env, seed=69, raises=False)

    # import matplotlib.pyplot as plt
    # import os
    #
    # img_path = "figures"
    # os.makedirs(img_path, exist_ok=True)
    #
    # X = env.get_wrapper_attr("observations")[-1].squeeze().T
    # U = env.get_wrapper_attr("actions")[-1].squeeze()
    # R = env.get_wrapper_attr("rewards")[-1]
    # T_R = env.get_wrapper_attr("rewards")
    # RWD = list(map(sum,T_R))
    # STP = list(map(len,T_R))
    #
    # _, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
    # axs[0].plot(X[0])
    # axs[1].plot(X[1])
    # axs[2].plot(X[2])
    # axs[3].plot(X[3])
    # axs[4].plot(U)
    # for i in range(2):
    #     # axs[0].axhline(env.get_wrapper_attr("x_bnd")[i][0], color="r")
    #     axs[2].axhline(env.get_wrapper_attr("x_bnd")[i][2], color="r")
    #     # axs[4].axhline(env.get_wrapper_attr("a_bnd")[i], color="r")
    # axs[0].set_ylabel("$X$")
    # axs[1].set_ylabel("$X'$")
    # axs[2].set_ylabel(r"$\theta$")
    # axs[3].set_ylabel(r"$\theta'$")
    # axs[4].set_ylabel("$a$")
    #
    # img_name = "state_last_episode.svg"
    # path = os.path.join(img_path, img_name)
    # plt.savefig(path, format="svg")
    #
    # plt.show()

