from typing import Literal, Optional, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from ..core.parameters import LearnableParametersDict, SymType
from ..core.schedulers import Scheduler
from .gradient_based_optimizer import GradientBasedOptimizer, LrType


class RMSprop(GradientBasedOptimizer[SymType, LrType]):
    """RMSprop optimizer, based on [1,2].

    References
    ----------
    [1] Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for
        machine learning lecture 6a overview of mini-batch gradient descent. page 14,
        2012.
    [2] RMSprop - PyTorch 2.1 documentation.
        https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    """

    _order = 1
    _hessian_sparsity = "diag"
    """In RMSprop, hessian is at most diagonal, i.e., in case we have constraints."""

    def __init__(
        self,
        learning_rate: Union[LrType, Scheduler[LrType]],
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        max_percentage_update: float = float("+inf"),
        bound_consistency: bool = False,
    ) -> None:
        """Instantiates the optimizer.

        Parameters
        ----------
        learning_rate : float/array, scheduler
            The learning rate of the optimizer. A float/array can be passed in case the
            learning rate must stay constant; otherwise, a scheduler can be passed which
            will be stepped `on_update` by default (see `hook` argument).
        alpha : float, optional
            A positive float that specifies the decay rate of the running average of the
            gradient. By default, it is set to `0.99`.
        eps : float, optional
            Term added to the denominator to improve numerical stability. By default, it
            is set to `1e-8`.
        weight_decay : float, optional
            A positive float that specifies the decay of the learnable parameters in the
            form of an L2 regularization term. By default, it is set to `0.0`, so no
            decay/regularization takes place.
        momentum : float, optional
            A positive float that specifies the momentum factor. By default, it is set
            to `0.0`, so no momentum is used.
        centered : bool, optional
            If `True`, compute the centered RMSProp, i.e., the gradient is normalized by
            an estimation of its variance.
        hook : {'on_update', 'on_episode_end', 'on_timestep_end'}, optional
            Specifies when to step the optimizer's learning rate's scheduler to decay
            its value (see `step` method also). This allows to vary the rate over the
            learning iterations. The options are:
             - `on_update` steps the learning rate after each agent's update
             - `on_episode_end` steps the learning rate after each episode's end
             - `on_timestep_end` steps the learning rate after each env's timestep.

            By default, 'on_update' is selected.
        max_percentage_update : float, optional
            A positive float that specifies the maximum percentage change the learnable
            parameters can experience in each update. For example,
            `max_percentage_update=0.5` means that the parameters can be updated by up
            to 50% of their current value. By default, it is set to `+inf`. If
            specified, the update becomes constrained and has to be solved as a QP,
            which is inevitably slower than its unconstrained counterpart.
        bound_consistency : bool, optional
            A boolean that, if `True`, forces the learnable parameters to lie in their
            bounds when updated. This is done `np.clip`. Only beneficial if numerical
            issues arise during updates, e.g., due to the QP solver not being able to
            guarantee bounds.
        """
        super().__init__(learning_rate, hook, max_percentage_update, bound_consistency)
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.centered = centered

    def set_learnable_parameters(self, pars: LearnableParametersDict[SymType]) -> None:
        super().set_learnable_parameters(pars)
        # initialize also running averages
        n = pars.size
        self._square_avg = np.zeros(n, dtype=float)
        self._grad_avg = np.zeros(n, dtype=float) if self.centered else None
        self._momentum_buf = np.zeros(n, dtype=float) if self.momentum > 0.0 else None

    def _first_order_update(
        self, gradient: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], Optional[str]]:
        theta = self.learnable_parameters.value

        # compute candidate update
        weight_decay = self.weight_decay
        lr = self.lr_scheduler.value
        if weight_decay > 0.0:
            gradient = gradient + weight_decay * theta
        dtheta, self._square_avg, self._grad_avg, self._momentum_buf = _rmsprop(
            gradient,
            self._square_avg,
            lr,
            self.alpha,
            self.eps,
            self.centered,
            self._grad_avg,
            self.momentum,
            self._momentum_buf,
        )

        # if unconstrained, apply the update directly; otherwise, solve the QP
        solver = self._update_solver
        if solver is None:
            return theta + dtheta, None
        lbx, ubx = self._get_update_bounds(theta)
        sol = solver(h=cs.DM.eye(theta.shape[0]), g=-dtheta, lbx=lbx, ubx=ubx)
        dtheta = np.asarray(sol["x"].elements())
        stats = solver.stats()
        return theta + dtheta, None if stats["success"] else stats["return_status"]


def _rmsprop(
    grad: np.ndarray,
    square_avg: np.ndarray,
    lr: LrType,
    alpha: float,
    eps: float,
    centered: bool,
    grad_avg: Optional[np.ndarray],
    momentum: float,
    momentum_buffer: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Computes the update's change according to Adam algorithm."""
    square_avg = alpha * square_avg + (1 - alpha) * np.square(grad)

    if centered:
        grad_avg = alpha * grad_avg + (1 - alpha) * grad
        avg = np.sqrt(square_avg - np.square(grad_avg))
    else:
        avg = np.sqrt(square_avg)
    avg += eps

    if momentum > 0.0:
        momentum_buffer = momentum * momentum_buffer + grad / avg
        dtheta = -lr * momentum_buffer
    else:
        dtheta = -lr * grad / avg
    return dtheta, square_avg, grad_avg, momentum_buffer
