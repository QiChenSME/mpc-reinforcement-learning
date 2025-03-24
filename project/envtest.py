import numpy as np
import gymnasium as gym

from CustomEnv.customtest import MassBlockEnv
from CustomEnv.CartpoleProMax import CartPoleV3, CartPoleVectorTest, CartPoleV4, CartPoleCS
import matplotlib.pyplot as plt
import keyboard

if __name__ == '__main__':
    steps = 100000
    act_data = []
    x_datas = []
    x_dot_datas = []
    theta_datas = []
    theta_dot_datas = []

    env = CartPoleCS(render_mode='human', ignore_terminal=True, tau=0.01)
    # env = gym.make('CartPole-v1', render_mode='human')+
    env.reset(seed=699)

    for i in range(steps):
        if keyboard.is_pressed('left'):
            action = -10.0
        elif keyboard.is_pressed('right'):
            action = 10.0
        else:
            action = 0
        action1 = np.array([1],dtype=np.float32)
        a = action

        data, _, _, _, _= env.step(a)

        act_data.append(a)
        x_datas.append(data[0])
        x_dot_datas.append(data[1])
        theta_datas.append(data[2])
        theta_dot_datas.append(data[3])

    plt.plot(act_data, label="action")
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(x_datas)
    axs[1].plot(x_dot_datas)
    axs[0].set_ylabel("$X$")
    axs[1].set_ylabel("$X'$")
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(theta_datas)
    axs[1].plot(theta_dot_datas)
    axs[0].set_ylabel(r"$\theta$")
    axs[1].set_ylabel(r"$\theta'$")
    # plt.legend(loc="best")

    plt.show()