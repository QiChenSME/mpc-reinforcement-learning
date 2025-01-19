import math
from typing import Optional

import casadi as cs
import gymnasium as gym
import numpy as np

from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

class CartPoleV3(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "render_fps": 100,
    }
    nx = 4
    nu = 1

    def __init__(
        self, *,
        render_mode: Optional[str] = None,
            gravity: float = 9.8,
            masscart: float = 1.0,
            masspole: float = 0.1,
            polelength: float = 0.5,
            tau: float = 0.01,
            x_threshold: float = 5.0,
            x_dot_threshold: float = 20,
            theta_threshold: float = 30,
            theta_dot_threshold: float = 1080 * 2 * math.pi / 360,
            force_threshold: float = 20.0,
            input_noise: float = 0.1,
            w: np.ndarray[np.float32] = np.asarray([[1e5], [1e2], [40], [1e2]]),
            ignore_terminal: bool = False,

    ):
        self.ignore_terminal = ignore_terminal
        self.time_step = 0

        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = polelength   # half the pole's length
        self.polemass_length = self.masspole * self.length
        self.tau = tau  # seconds between state updates
        self.kinematics_integrator = "semi-implicit euler"

        # Angle at which to fail the episode
        self.theta_threshold = theta_threshold
        self.theta_threshold_radians = theta_threshold * 2 * math.pi / 360
        self.theta_dot_threshold = theta_dot_threshold
        self.x_threshold = x_threshold
        self.x_dot_threshold = x_dot_threshold
        self.force_threshold = force_threshold

        self.x_bnd = (np.asarray([[-x_threshold*0.8],
                                  [-x_dot_threshold],
                                  [-self.theta_threshold_radians],
                                  [-theta_dot_threshold]]),
                      np.asarray([[x_threshold*0.8],
                                  [x_dot_threshold],
                                  [self.theta_threshold_radians],
                                  [theta_dot_threshold]]))
        self.a_bnd = (-force_threshold, force_threshold)
        self.w = w
        self.e_bnd = (-input_noise, input_noise)

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        # 定义边界数值（此处定义为正方向一侧的边界值）
        high = np.array(
            [
                self.x_threshold,
                self.x_dot_threshold,
                self.theta_threshold_radians * 2,
                self.theta_dot_threshold,
            ],
            dtype=np.float32,
        )

        # 定义动作空间及状态空间（此处命名为观测空间）
        self.action_space = spaces.Box(-self.force_threshold, self.force_threshold, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # 观测空间取多维连续空间，high规定了上下界

        # 渲染器的相关设置
        self.render_mode = render_mode

        self.screen_width = 1600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.force_record = None
        self.reward_record = None

        # 清空越界判定
        self.steps_beyond_terminated = None


    def step(self,
             action:cs.DM,
             ):
        # 检查是否reset
        # 注意此处产生的报错，排查不能通过检查的原因
        force = float(action)
        # assert self.action_space.contains(
        #     action
        # ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        # 从实例的state属性中获取环境的状态数据
        x, x_dot, theta, theta_dot = self.state
        if x <= -self.x_threshold and force < 0:
            force = 0
        elif x >= self.x_threshold and force > 0:
            force = 0
        if x <= -self.x_threshold and x_dot < 0:
            x_dot = 0
        elif x >= self.x_threshold and x_dot > 0:
            x_dot = 0

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        # 计算输入的实际物理量
        # 中间变量
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        # 角加速度
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        # 加速度
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # 更新状态
        # 欧拉积分
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        # 半隐式欧拉积分
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        # 将更新后的状态数据放入nparray中，赋值给state属性
        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64).reshape(4,1)
        # 判断是否终止
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        # 当没有终止时给予奖励（或不给予负奖励）
        if not terminated:
            lb, ub = self.x_bnd[0]/2, self.x_bnd[1]/2
            reward = float(
                0.5
                * (
                    0.1*x_dot**2 + 1*x**2 + 2*theta**2 + 0.2*theta_dot**2
                    + 0.1 * action ** 2
                    + self.w.T @ np.maximum(0, lb - self.state)
                    + self.w.T @ np.maximum(0, self.state - ub)
                )
            )
        # 若达到终止且终止判定未更新，更新终止判定
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 100000
        # 终止判定后仍调用step则抛出警告
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
        # 给予零奖励
            reward = 100000

        self.time_step += 1

        # 判断是否渲染
        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        if self.ignore_terminal:
            return np.array(self.state, dtype=np.float32), reward, False, False, {}
        else:
            return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.15, 0.15  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,1))
        self.steps_beyond_terminated = None
        self.time_step = 0
        self.force_record = 0
        self.reward_record = np.zeros(7)

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}


    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        font = pygame.font.Font(None, 20)

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 8.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        pos = font.render(f'X: {self.state[0][0]:.2f}', True, (0, 0, 0))
        thetaD = font.render(f'ThetaD: {(self.state[2][0]/np.pi*180):.2f}', True, (0, 0, 0))
        theta = font.render(f'Theta: {self.state[2][0]:.2f}', True, (0, 0, 0))
        vol = font.render(f'V: {self.state[1][0]:.2f}', True, (0, 0, 0))
        thetaV = font.render(f'ThetaV: {self.state[3][0]:.2f}', True, (0, 0, 0))
        time = font.render(f'time: {(self.time_step*self.tau):.2f}', True, (0, 0, 0))
        force = font.render(f'force: {self.force_record :.2f}', True, (0, 0, 0))

        reward = font.render(f'loss: {self.reward_record[0] :.2f}', True, (0, 0, 0))
        reward1 = font.render(f'x loss: {self.reward_record[1] :.2f}', True, (0, 0, 0))
        reward2 = font.render(f'theta loss: {self.reward_record[2] :.2f}', True, (0, 0, 0))
        reward3 = font.render(f'xdot loss: {self.reward_record[3] :.2f}', True, (0, 0, 0))
        reward4 = font.render(f'thetadot loss: {self.reward_record[4] :.2f}', True, (0, 0, 0))
        reward5 = font.render(f'boundary loss: {self.reward_record[5] :.2f}', True, (0, 0, 0))
        reward6 = font.render(f'action loss: {self.reward_record[6] :.2f}', True, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        self.screen.blit(pos, (0, 0))
        self.screen.blit(thetaD, (0, 11))
        self.screen.blit(theta, (0, 23))
        self.screen.blit(vol, (0, 35))
        self.screen.blit(thetaV, (0, 47))
        self.screen.blit(time, (0, 59))
        self.screen.blit(force, (0, 83))

        self.screen.blit(reward, (119, 0))
        self.screen.blit(reward1, (119, 11))
        self.screen.blit(reward2, (119, 23))
        self.screen.blit(reward3, (119, 35))
        self.screen.blit(reward4, (119, 47))
        self.screen.blit(reward5, (119, 59))
        self.screen.blit(reward6, (119, 71))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class CartPoleVectorTest(CartPoleV3):
    def __init__(self, **kwargs):
        super(CartPoleVectorTest, self).__init__(**kwargs)
        M = self.masscart
        m = self.masspole
        g = self.gravity
        l = self.length
        Ts = self.tau
        C = M * m * (l**2)
        self.A = np.eye(4) + np.asarray([
            [0, 1, 0, 0],
            [0, 0, (- (m**3) * (l**4) * g * (M+m) + (m**4) * (g**2) * (l**4))/C, 0],
            [0, 0, 0, 1],
            [0, 0, (M+m)*m*g*l, 0]]) * Ts / C
        self.B = np.asarray([
            [0],
            [m * (l**2)],
            [0],
            [-m * l]]) * Ts / C

    def step(self, action):
        force = float(action)
        new = self.A @ self.state + self.B @ np.asarray([force]).reshape(1,1)
        self.state = np.asarray(new).reshape(4,1)

        reward = 0.0

        terminated = False

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, **kwargs):
        super(CartPoleVectorTest, self).reset(**kwargs)
        self.state = np.asarray([[0],[0],[np.pi/4],[0]])
        return np.array(self.state)


class CartPoleV4(CartPoleV3):
    def __init__(self, **kwargs):
        super(CartPoleV4, self).__init__(**kwargs)
        # 定义边界数值（此处定义为正方向一侧的边界值）
        self.theta_threshold = 180
        self.theta_threshold_radians = np.pi
        self.x_bnd[0][2][0] = -self.theta_threshold_radians
        self.x_bnd[1][2][0] = +self.theta_threshold_radians
        self.x_bnd[0][0][0] = -self.x_threshold
        self.x_bnd[1][0][0] = +self.x_threshold
        high = np.array(
            [
                self.x_threshold,
                self.x_dot_threshold,
                np.pi,
                self.theta_dot_threshold,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)


    def step(self,
             action: cs.DM,
             ):
        # 检查是否reset
        # 注意此处产生的报错，排查不能通过检查的原因
        force = float(action)
        self.force_record = force
        force += self.np_random.uniform(*self.e_bnd)
        # assert self.action_space.contains(
        #     action
        # ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        # 从实例的state属性中获取环境的状态数据
        x, x_dot, theta, theta_dot = self.state
        # if x <= -self.x_threshold and force < 0:
        #     force = 0
        # elif x >= self.x_threshold and force > 0:
        #     force = 0
        if x <= -self.x_threshold:
            force += (-self.x_threshold-x)*20000
        elif x >= self.x_threshold:
            force -= (x-self.x_threshold)*20000

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        # 计算输入的实际物理量
        # 中间变量
        temp = (
                       force + self.polemass_length * np.square(theta_dot) * sintheta
               ) / self.total_mass
        # 角加速度
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        # 加速度
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        # if x <= -self.x_threshold and force == 0:
        #     xacc += float(action)
        #     if xacc < 0: xacc = 0
        # elif x >= self.x_threshold and force == 0:
        #     xacc += float(action)
        #     if xacc > 0: xacc = 0

        # 更新状态
        # 欧拉积分
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        # 半隐式欧拉积分
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        while theta <= -np.pi*1.5:
            theta += 2 * np.pi
        while theta > np.pi*1.5:
            theta -= 2 * np.pi
        if x <= -self.x_threshold and x_dot < 0:
            x_dot = [0.0]
        elif x >= self.x_threshold and x_dot > 0:
            x_dot = [0.0]
        x = np.clip(x, -self.x_threshold-0.003, self.x_threshold+0.003)
        x_dot = np.clip(x_dot, -self.x_dot_threshold, self.x_dot_threshold)
        theta_dot = np.clip(theta_dot, -self.theta_dot_threshold, self.theta_dot_threshold)

        # 将更新后的状态数据放入nparray中，赋值给state属性
        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64).reshape(4, 1)

        lb, ub = self.x_bnd[0], self.x_bnd[1]
        reward = float(
            0.5
            * (
                    0.1 * x_dot ** 2 + 1 * x ** 2 + 20 * math.sqrt(math.sqrt(theta ** 2)) + 0.2 * theta_dot ** 2
                    + 0.01 * action ** 2
                    + self.w.T @ np.maximum(0, lb - self.state)
                    + self.w.T @ np.maximum(0, self.state - ub)
            )
        )

        self.time_step += 1
        self.reward_record[0] = reward
        self.reward_record[1] = 0.5 * 1 * x ** 2
        self.reward_record[2] = 0.5 * 20 * math.sqrt(math.sqrt(theta ** 2))
        self.reward_record[3] = 0.5 * 0.1 * x_dot ** 2
        self.reward_record[4] = 0.5 * 0.2 * theta_dot ** 2
        self.reward_record[6] = 0.5 * 0.01 * action ** 2
        self.reward_record[5] = 0.5 * (self.w.T @ np.maximum(0, lb - self.state)
                                 + self.w.T @ np.maximum(0, self.state - ub))

        # 判断是否渲染
        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, False, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.5, 0.5  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,1))
        self.state[2][0] = np.pi
        self.steps_beyond_terminated = None
        self.time_step = 0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}
