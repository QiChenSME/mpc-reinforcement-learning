import casadi as cs
from csnlp import Nlp
from csnlp.wrappers import Mpc

from mpcrl.util.control import dlqr
from mpcrl.core.exploration import *

from Env import StageEnv
from Dynamics import WaferStage

class NonLinearMpc(Mpc[cs.SX]):
    """A simple nonlinear MPC controller."""
    model = WaferStage()
    env = StageEnv(model, render_mode='human')
    env.reset()

    horizon = 5
    discount_factor = 0.9

    A_init, B_init = env.jacobian(env.state.flatten(), np.zeros(env.nu))
    A_init = A_init.full().reshape((env.nx, env.nx))
    B_init = B_init.full().reshape((env.nx, env.nu))

    learnable_pars_init = {
        "V0": np.asarray(0.0),
        "x_lb": np.asarray(env.x_bnd[0]).reshape(env.nx, ),
        "x_ub": np.asarray(env.x_bnd[1]).reshape(env.nx, ),
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
    env.close()

    def __init__(self, *args, **kwargs) -> None:
        N = self.horizon
        gamma = self.discount_factor
        w = self.env.w_x
        nx, nu = self.env.nx, self.env.nu
        x_bnd, a_bnd = self.env.x_bnd, self.env.u_bnd
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
            V0
            + cs.bilin(S, x[:, -1])/N
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers * (0
                        + cs.sum1(Q @ x[:, :-1] * x[:, :-1])
                        + 0.5 * cs.sum1(R @ u * u)/N
                        + w.T @ s
                )
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


class LinearMpc(Mpc[cs.SX]):
    """Linear MPC controller."""
    model = WaferStage()
    env = StageEnv(model, render_mode='human')
    env.reset()

    horizon = 5
    discount_factor = 0.9

    A_init, B_init = env.jacobian(env.state.flatten(), np.zeros(env.nu))
    A_init = A_init.full().reshape((env.nx, env.nx))
    B_init = B_init.full().reshape((env.nx, env.nu))

    A_offset = np.zeros((env.nx, env.nx))
    B_offset = np.zeros((env.nx, env.nu))
    A_proportion = np.eye(env.nx)
    B_proportion = np.eye(env.nx)

    learnable_pars_init = {
        "V0": np.asarray(0.0),
        "x_lb": np.asarray(env.x_bnd[0]).reshape(env.nx, ),
        "x_ub": np.asarray(env.x_bnd[1]).reshape(env.nx, ),
        "f": np.zeros(env.nx + env.nu),
        "Q": 10 * np.eye(env.nx),
        "R": 0.1 * np.eye(env.nu),
        "A_o": A_offset,
        "B_o": B_offset,
        "A_p": A_proportion,
        "B_p": B_proportion,
    }
    fixed_pars_init = {
        "S": dlqr(A_init,
                  B_init,
                  learnable_pars_init["Q"],
                  learnable_pars_init["R"])[1],
        "A": A_init,
        "B": B_init,
        "b": np.zeros(env.nx),
    }
    env.close()

    def __init__(self, *args, **kwargs) -> None:
        N = self.horizon
        gamma = self.discount_factor
        w = self.env.w_x
        nx, nu = self.env.nx, self.env.nu
        x_bnd, a_bnd = self.env.x_bnd, self.env.u_bnd
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

        A_o = self.parameter("A_o", (nx, nx))
        B_o = self.parameter("B_o", (nx, nu))
        A_p = self.parameter("A_p", (nx, nx))
        B_p = self.parameter("B_p", (nx, nx))

        A = self.parameter("A", (nx, nx))
        B = self.parameter("B", (nx, nu))
        b = self.parameter("b", (nx, 1))

        # variables (state, action, slack)
        x, _ = self.state("x", nx, bound_initial=False)
        u, _ = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s, _, _ = self.variable("s", (nx, N), lb=0)

        # dynamics
        self.set_affine_dynamics(A+A_o, B+B_o, c=b)

        # other constraints
        self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)

        # objective

        # A_init, B_init = self.fixed_pars_init["A"], self.fixed_pars_init["B"]
        S_init = cs.DM(dlqr(self.A_init, self.B_init, 0.5 * np.eye(nx), 0.25 * np.eye(nu))[1])

        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            V0
            + cs.bilin(S, x[:, -1])
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers * (0
                        + cs.sum1(Q @ x[:, :-1] * x[:, :-1])
                        + 0 * cs.sum1(R @ u * u)
                        + w.T @ s
                )
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


if __name__ == "__main__":
    mpc = NonLinearMpc()
