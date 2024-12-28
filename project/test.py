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

from CustomEnv.customtest import MassBlockEnv


class LinearMpc(Mpc[cs.SX]):
    """A simple linear MPC controller."""

    horizon = 10
    discount_factor = 0.9
    learnable_pars_init = {
        "V0": np.asarray(0.0),
        "x_lb": np.asarray([-10, -10]),
        "x_ub": np.asarray([10, 10]),
        "b": np.zeros(MassBlockEnv.nx),
        "f": np.zeros(MassBlockEnv.nx + MassBlockEnv.nu),
        "A": np.asarray([[1, 0.25], [0, 1]]),
        "B": np.asarray([[0.0312], [0.25]]),
    }

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        w = MassBlockEnv.w
        nx, nu = MassBlockEnv.nx, MassBlockEnv.nu
        x_bnd, a_bnd = MassBlockEnv.x_bnd, MassBlockEnv.a_bnd
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
    env = MonitorEpisodes(TimeLimit(MassBlockEnv(), max_episode_steps=2_00))

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
    agent = Log(
        RecordUpdates(
            LstdQLearningAgent(
                mpc=mpc,
                learnable_parameters=learnable_pars,
                discount_factor=mpc.discount_factor,
                update_strategy=1,
                optimizer=NetwonMethod(learning_rate=5e-2),
                hessian_type="approx",
                record_td_errors=True,
                remove_bounds_on_initial_action=True,
            )
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 1000},
    )

    # launch the training simulation
    agent.train(env=env, episodes=100, seed=69)

    # plot the results
    import matplotlib.pyplot as plt

    X = env.get_wrapper_attr("observations")[0].squeeze().T
    U = env.get_wrapper_attr("actions")[0].squeeze()
    R = env.get_wrapper_attr("rewards")[0]
    _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    axs[0].plot(X[0])
    axs[1].plot(X[1])
    axs[2].plot(U)
    for i in range(2):
        axs[0].axhline(env.get_wrapper_attr("x_bnd")[i][0], color="r")
        axs[1].axhline(env.get_wrapper_attr("x_bnd")[i][1], color="r")
        axs[2].axhline(env.get_wrapper_attr("a_bnd")[i], color="r")
    axs[0].set_ylabel("$s_1$")
    axs[1].set_ylabel("$s_2$")
    axs[2].set_ylabel("$a$")

    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(agent.td_errors, "o", markersize=1)
    axs[1].semilogy(R, "o", markersize=1)
    axs[0].set_ylabel(r"$\tau$")
    axs[1].set_ylabel("$L$")

    _, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
    axs[0, 0].plot(np.asarray(agent.updates_history["b"]))
    axs[0, 1].plot(
        np.stack(
            [np.asarray(agent.updates_history[n])[:, 0] for n in ("x_lb", "x_ub")], -1
        ),
    )
    axs[1, 0].plot(np.asarray(agent.updates_history["f"]))
    axs[1, 1].plot(np.asarray(agent.updates_history["V0"]))
    axs[2, 0].plot(np.asarray(agent.updates_history["A"]).reshape(-1, 4))
    axs[2, 1].plot(np.asarray(agent.updates_history["B"]).squeeze())
    axs[0, 0].set_ylabel("$b$")
    axs[0, 1].set_ylabel("$x_1$")
    axs[1, 0].set_ylabel("$f$")
    axs[1, 1].set_ylabel("$V_0$")
    axs[2, 0].set_ylabel("$A$")
    axs[2, 1].set_ylabel("$B$")
    plt.show()

    print("b:", agent.updates_history["b"][-1])
    print("x1:", np.stack(
            [np.asarray(agent.updates_history[n])[:, 0] for n in ("x_lb", "x_ub")], -1
        )[-1])
    print("f:", agent.updates_history["f"][-1])
    print("V0:", agent.updates_history["V0"][-1])
    print("A:", agent.updates_history["A"][-1])
    print("B:", agent.updates_history["B"][-1])
