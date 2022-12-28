from typing import (
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    SupportsFloat,
    Tuple,
    Union,
)

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc, NlpSensitivity
from scipy.linalg import cho_solve
from typing_extensions import TypeAlias

from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.agents.learning_agents import RlLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.util.math import cholesky_added_multiple_identities
from mpcrl.util.types import GymEnvLike

ExpType: TypeAlias = Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]


class LstdQLearningAgent(RlLearningAgent[SymType, ExpType]):
    """Second-order Least-Squares Temporal Difference (LSTD) Q-learning agent, as first
    proposed in a simpler format in [1], and then in [2].

    The Q-learning agent uses an MPC controller as policy provider and function
    approximation, and adjusts its parametrization according to the temporal-difference
    error, with the goal of improving the policy, in an indirect fashion by learning the
    action value function.

    References
    ----------
    [1] Gros, S. and Zanon, M., 2019. Data-driven economic NMPC using reinforcement
        learning. IEEE Transactions on Automatic Control, 65(2), pp. 636-648.
    [2] Esfahani, H.N., Kordabad, A.B. and Gros, S., 2021, June. Approximate Robust NMPC
        using Reinforcement Learning. In 2021 European Control Conference (ECC), pp.
        132-137. IEEE.
    """

    __slots__ = ("_dQdtheta", "_d2Qdtheta2", "td_errors", "chol_maxiter")

    def __init__(
        self,
        mpc: Mpc[SymType],
        discount_factor: float,
        learning_rate: Union[Scheduler[npt.NDArray[np.double]], npt.NDArray[np.double]],
        learnable_parameters: LearnableParametersDict[SymType],
        fixed_parameters: Union[
            None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Optional[ExperienceReplay[ExpType]] = None,
        experience_sample_size: Union[int, float] = 1,
        experience_sample_include_last: Union[int, float] = 0,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        stepping: Literal["on_update", "on_episode_start", "on_env_step"] = "on_update",
        hessian_type: Literal["approx", "full"] = "approx",
        record_td_errors: bool = False,
        chol_maxiter: int = 1000,
        name: Optional[str] = None,
    ) -> None:
        """Instantiates the learning agent.

        Parameters
        ----------
        mpc : Mpc[casadi.SX or MX]
            The MPC controller used as policy provider by this agent. The instance is
            modified in place to create the approximations of the state function `V(s)`
            and action value function `Q(s,a)`, so it is recommended not to modify it
            further after initialization of the agent. Moreover, some parameter and
            constraint names will need to be created, so an error is thrown if these
            names are already in use in the mpc. These names are under the attributes
            `perturbation_parameter`, `action_parameter` and `action_constraint`.
        discount_factor : float
            In RL, the factor that discounts future rewards in favor of immediate
            rewards. Usually denoted as `\\gamma`. Should be a number in (0, 1).
        learning_rate : Scheduler of array
            The learning rate of the algorithm, in general, a small number. A scheduler
            can be passed so that the learning rate is decayed after every step (see
            `stepping`).
        learnable_parameters : LearnableParametersDict
            A special dict containing the learnable parameters of the MPC, together with
            their bounds and values. This dict is complementary with `fixed_parameters`,
            which contains the MPC parameters that are not learnt by the agent.
        fixed_parameters : dict[str, array_like] or collection of, optional
            A dict (or collection of dict, in case of `csnlp.MultistartNlp`) whose keys
            are the names of the MPC parameters and the values are their corresponding
            values. Use this to specify fixed parameters, that is, non-learnable. If
            `None`, then no fixed parameter is assumed.
        exploration : ExplorationStrategy, optional
            Exploration strategy for inducing exploration in the MPC policy. By default
            `None`, in which case `NoExploration` is used in the fixed-MPC agent.
        experience : ExperienceReplay, optional
            The container for experience replay memory. If `None` is passed, then a
            memory wtih length 1 is created, i.e., it keeps only the latest memoery
            transition.
        experience_sample_size : int or float, optional
            Size (or percentage of replay `maxlen`) of the experience replay items to
            draw when performing an update. By default, one item per sampling is drawn.
        experience_sample_include_last : int or float, optional
            Size (or percentage of sample size) dedicated to including the latest
            experience transitions. By default, 0, i.e., no last item is included.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        stepping : {'on_update', 'on_episode_start', 'on_env_step'}, optional
            Specifies to the algorithm when to step its schedulers (e.g., for learning
            rate and/or exploration decay), either after
             - each agent's update ('agent-update')
             - each episode's start ('ep-start')
             - each environment's step ('env-step').
            By default, 'on_update' is selected.
        hessian_type : 'approx' or 'full', optional
            The type of hessian to use in this second-order algorithm. If `approx`, an
            easier approximation of it is used; otherwise, the full hessian is computed
            but this is much more expensive.
        record_td_errors: bool, optional
            If `True`, the TD errors are recorded in the field `td_errors`, which
            otherwise is `None`. By default, does not record them.
        chol_maxiter : int, optional
            Minor setting to change to maximum number of iterations in the Cholesky's
            factorization with additive multiples of the identity to ensure positive
            definiteness of the hessian. By default, 1000.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        super().__init__(
            mpc=mpc,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            learnable_parameters=learnable_parameters,
            fixed_parameters=fixed_parameters,
            exploration=exploration,
            experience=experience,
            experience_sample_size=experience_sample_size,
            experience_sample_include_last=experience_sample_include_last,
            warmstart=warmstart,
            stepping=stepping,
            name=name,
        )
        self._dQdtheta, self._d2Qdtheta2 = self._init_Q_derivatives(hessian_type)
        self.chol_maxiter = chol_maxiter
        self.td_errors: Optional[List[float]] = [] if record_td_errors else None

    @property
    def learning_rate(self) -> npt.NDArray[np.double]:
        """Gets the learning rate of the Q-learning agent."""
        return self._learning_rate_scheduler.value

    def step(self) -> None:
        """Steps the learning rate and exploration strength/chance for the agent
        (usually, these decay over time)."""
        self._learning_rate_scheduler.step()
        super().step()

    def store_experience(  # type: ignore
        self, cost: SupportsFloat, solQ: Solution[SymType], solV: Solution[SymType]
    ) -> None:
        """Stores the gradient and hessian for the current transition in memoru.

        Parameters
        ----------
        cost : float
            The cost of this state transition.
        solQ : Solution[SymType]
            The solution to `Q(s,a)`.
        solV : Solution[SymType]
            The solution to `V(s+)`.
        """
        inp: cs.DM = solQ._get_value.keywords["new"]
        dQ = self._dQdtheta(inp).full().reshape(-1, 1)
        ddQ = self._d2Qdtheta2(inp).full()
        td_error = cost + self.discount_factor * solV.f - solQ.f
        g = -td_error * dQ
        H = dQ @ dQ.T - td_error * ddQ
        if self.td_errors is not None:
            self.td_errors.append(td_error)
        return super().store_experience((g, H))

    def update(self) -> Optional[str]:
        lr = self._learning_rate_scheduler.value
        sample = self.sample_experience()
        g, H = (np.mean(tuple(o), axis=0) for o in zip(*sample))
        R = cholesky_added_multiple_identities(H, maxiter=self.chol_maxiter)
        p = lr * cho_solve((R, True), g, check_finite=False).reshape(-1)

        theta = self._learnable_pars.value  # current values of parameters
        solver = self._update_solver
        if solver is None:
            self._learnable_pars.update_values(theta - p)
            return None

        sol = solver(
            p=np.concatenate((theta, p)),
            lbx=self._learnable_pars.lb,
            ubx=self._learnable_pars.ub,
            x0=theta - p,
        )
        self._learnable_pars.update_values(sol["x"].full().reshape(-1))
        stats = solver.stats()
        return None if stats["success"] else stats["return_status"]

    @staticmethod
    def train_one_episode(
        agent: "LstdQLearningAgent[SymType]",
        env: GymEnvLike[ObsType, ActType],
        episode: int,
        init_state: ObsType,
        update_cycle: Iterator[bool],
        raises: bool = True,
    ) -> float:
        truncated = terminated = False
        timestep = rewards = 0
        state = init_state

        # solve for the first action
        action, solV = agent.state_value(state, deterministic=False)
        if not solV.success:
            agent.on_mpc_failure(episode, -1, solV.status, raises)

        while not (truncated or terminated):
            # compute Q(s,a)
            solQ = agent.action_value(state, action)

            # step the system with action computed at the previous iteration
            state, r, truncated, terminated, _ = env.step(action)
            agent.on_env_step(env, episode, timestep)

            # compute V(s+)
            action, solV = agent.state_value(state, deterministic=False)
            if solQ.success and solV.success:
                agent.store_experience(r, solQ, solV)
            else:
                agent.on_mpc_failure(episode, timestep, solV.status, raises)

            # check if it is time to update
            if next(update_cycle):
                if update_msg := agent.update():
                    agent.on_update_failure(episode, timestep, update_msg, raises)
                agent.on_update()

            # increase counters
            rewards += r  # type: ignore
            timestep += 1
        return rewards

    def _init_Q_derivatives(
        self, hessian_type: Literal["approx", "full"]
    ) -> Tuple[cs.Function, cs.Function]:
        """Internal utility to compute the derivative of Q(s,a) w.r.t. the learnable
        parameters, a.k.a., theta."""
        theta = cs.vertcat(*self._learnable_pars.sym.values())
        nlp = NlpSensitivity(self._Q.nlp, target_parameters=theta)
        Lt = nlp.jacobians["L-p"]  # a.k.a., dQdtheta
        Ltt = nlp.hessians["L-pp"]  # a.k.a., approximated d2Qdtheta2
        if hessian_type == "approx":
            d2Qdtheta2 = Ltt
        elif hessian_type == "full":
            dydtheta, _ = nlp.parametric_sensitivity(second_order=False)
            d2Qdtheta2 = dydtheta.T @ nlp.jacobians["K-p"] + Ltt
        else:
            raise ValueError(f"Invalid type of hessian; got {hessian_type}.")

        # convert to functions (much faster runtime)
        inp = cs.vertcat(nlp.p, nlp.x, nlp.lam_g, nlp.lam_h, nlp.lam_lbx, nlp.lam_ubx)
        dQdtheta_ = cs.Function("dQdtheta", [inp], [Lt])
        d2Qdtheta2_ = cs.Function("d2Qdtheta2", [inp], [d2Qdtheta2])
        assert (
            not dQdtheta_.has_free() and not d2Qdtheta2_.has_free()
        ), "Internal error in Q derivatives."
        return dQdtheta_, d2Qdtheta2_


# TODO:
# max update percentage
