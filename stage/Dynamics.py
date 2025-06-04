import time
from typing import Optional

import casadi as cs
import numpy as np


class WaferStage:
    _nx = 12
    _nu = 6
    def __init__(self,
                 time_step:Optional[float] = 0.001,
                 m:Optional[float] = 5.0,
                 g:Optional[float] = 9.81,
                 j_xx:Optional[float] = 0.033,
                 j_yy:Optional[float] = 0.0165,
                 j_zz:Optional[float] = 0.066,
                 integrator_type:Optional[str] = "cvodes",
                 ):
        # 系统量
        self.time_step = 0.001
        # 常量
        self.m = m
        self.g = g
        self.J_xx = j_xx
        self.J_yy = j_yy
        self.J_zz = j_zz
        # 符号变量
        # 输入量
        F_x = cs.MX.sym("F_x")  # 平移力x
        F_y = cs.MX.sym("F_y")  # 平移力y
        F_z = cs.MX.sym("F_z")  # 平移力z
        M_x = cs.MX.sym("M_x")  # 转矩x
        M_y = cs.MX.sym("M_y")  # 转矩y
        M_z = cs.MX.sym("M_z")  # 转矩z
        # 零阶量
        x = cs.MX.sym("x")  # 位移x
        y = cs.MX.sym("y")  # 位移y
        z = cs.MX.sym("z")  # 位移z
        theta_x = cs.MX.sym("theta_x")  # 角度x
        theta_y = cs.MX.sym("theta_y")  # 角度y
        theta_z = cs.MX.sym("theta_z")  # 角度z
        # 一阶量
        v_x = cs.MX.sym("v_x")  # 平移速度x
        v_y = cs.MX.sym("v_y")  # 平移速度y
        v_z = cs.MX.sym("v_z")  # 平移速度z
        omega_x = cs.MX.sym("omega_x")  # 角速度x
        omega_y = cs.MX.sym("omega_y")  # 角速度y
        omega_z = cs.MX.sym("omega_z")  # 角速度z
        # 二阶量
        acc_x = F_x / m
        acc_y = F_y / m
        acc_z = F_z / m - g
        alpha_x = (M_x + (self.J_yy - self.J_zz) * omega_y * omega_z) / self.J_xx
        alpha_y = (M_y + (self.J_zz - self.J_xx) * omega_z * omega_x) / self.J_yy
        alpha_z = (M_z + (self.J_xx - self.J_yy) * omega_x * omega_y) / self.J_zz
        # 微分方程定义
        dx_dt = v_x
        dy_dt = v_y
        dz_dt = v_z
        dtheta_x_dt = omega_x
        dtheta_y_dt = omega_y
        dtheta_z_dt = omega_z
        dv_x_dt = acc_x
        dv_y_dt = acc_y
        dv_z_dt = acc_z
        domega_x_dt = alpha_x
        domega_y_dt = alpha_y
        domega_z_dt = alpha_z
        # 配置积分器
        state = cs.vertcat(x, y, z,
                           theta_x, theta_y, theta_z,
                           v_x, v_y, v_z,
                           omega_x, omega_y, omega_z)
        params = cs.vertcat(F_x, F_y, F_z, M_x, M_y, M_z)
        rhs = cs.vertcat(dx_dt, dy_dt, dz_dt, dtheta_x_dt, dtheta_y_dt, dtheta_z_dt,
                         dv_x_dt, dv_y_dt, dv_z_dt, domega_x_dt, domega_y_dt, domega_z_dt)
        ode_system = {
            "x": state,
            "p": params,  # 输入量
            "ode": rhs
        }
        self._integrator = cs.integrator('F', integrator_type, ode_system, 0, time_step)

        # 抽象符号
        self._sym_x = state
        self._sym_u = params

        A = cs.jacobian(rhs, self._sym_x) * self.time_step + np.eye(self._nx)
        B = cs.jacobian(rhs, self._sym_u) * self.time_step

        self._jacobian= cs.Function('jacobian', [self._sym_x, self._sym_u], [A, B])

    @property
    def integrator(self) -> cs.Function:
        return self._integrator

    @property
    def jacobian(self) -> cs.Function:
        return self._jacobian

    @property
    def sym_x(self) -> cs.MX:
        return self._sym_x

    @property
    def sym_u(self) -> cs.MX:
        return self._sym_u

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def nu(self) -> int:
        return self._nu


if __name__ == '__main__':
    from matplotlib.animation import FuncAnimation

    stage = WaferStage()
    dynamics = stage.integrator
    jacobian = stage.jacobian
    # 调用积分器进行计算
    state = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    start_time = time.time()
    for i in range(1000):
        sys_input = np.random.uniform(low=-1, high=1, size=6)
        result = dynamics(x0=state, p=sys_input)
        print(f"step: {i} system input: {sys_input} result: {result['xf']}")
        A, B = jacobian(state, sys_input)
        A = A.full().reshape(12, 12)
        B = B.full().reshape(12, 6)
        print(f"A: {A},\n B: {B}")
        state = result['xf']
    end_time = time.time()
    print(f"total time: {end_time - start_time}")


