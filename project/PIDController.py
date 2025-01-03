from enum import Enum, auto
import numpy as np

class InputType(Enum):
    SINGLE = auto()
    DUAL = auto()

class FeedbackMethod(Enum):
    Positive = auto()
    Negative = auto()

class PIDController:
    def __init__(self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.5,
        tau: float = 0.01,
        u_bud: np.ndarray = None,
        input_type: InputType = InputType.SINGLE,
        feedback_method: FeedbackMethod = FeedbackMethod.Positive,
         ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tau = tau
        if u_bud is None:
            self.u_bud = np.asarray(([-np.inf, np.inf]), dtype=np.float64)
        elif u_bud.shape != (2,):
            raise ValueError('u_bud must be an array of shape (2, 1)')
        elif u_bud[0]>u_bud[1]:
            raise ValueError('u_bud must be strictly increasing')
        else:
            self.u_bud = u_bud
        self.input_type = input_type
        self.feedback_method = feedback_method
        self.integral = 0
        self.difference = 0
        self.last_err = None

    def reset(self, tar = 0,x = 0):
        self.integral = 0
        self.difference = 0
        self.last_err = tar - x

    def pid(self, tar, x, x_dot = None):
        err = tar - x

        if self.last_err is None:
            raise Warning('\'D\' will not work since last_err is None, call reset() before pid()')
        else:
            self.difference = (err - self.last_err) / self.tau
        self.integral += err * self.tau

        output = 0
        if self.input_type == InputType.SINGLE:
            output = self.kp * err + self.ki * self.integral - self.kd * self.difference
        elif self.input_type == InputType.DUAL:
            if x_dot is None:
                raise Warning('PID will not work since the feedback method is \'DUAL\', but x_dot is \'None\'')
            output = self.kp * err + self.ki * self.integral - self.kd * x_dot

        if self.feedback_method == FeedbackMethod.Positive:
            return output
        elif self.feedback_method == FeedbackMethod.Negative:
            return -output
        else:
            return 0

