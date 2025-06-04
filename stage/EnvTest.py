import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from Env import StageEnv
from Dynamics import WaferStage

model = WaferStage(g=0)
env = StageEnv(model, render_mode='human')

if __name__ == '__main__':
    steps = 10000
    env.reset(seed=699)

    for j in range(3):
        for i in range(steps):
            # env.step(env.action_space.sample())
            env.step(np.zeros((env.nu, 1)))
            if not plt.fignum_exists(env.fig.number):  # 检查图形是否存在
                break  # 如果图形窗口被关闭，退出循环
        if not plt.fignum_exists(env.fig.number):  # 检查图形是否存在
            print("图形窗口已关闭，程序退出。")
            break  # 如果图形窗口被关闭，退出循环
        env.reset()