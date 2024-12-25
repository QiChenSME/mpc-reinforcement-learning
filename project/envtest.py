import numpy as np

from CustomEnv.customtest import MassBlockEnv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    steps = 100
    act_data = []
    x_datas = []
    x_dot_datas = []

    env = MassBlockEnv()
    env.reset()

    for i in range(steps):
        action = env.action_space.sample()
        action1 = np.array([1],dtype=np.float32)
        data, _, _, _, _= env.step(action1)
        act_data.append(action1)
        x_datas.append(data[0])
        x_dot_datas.append(data[1])

    plt.plot(act_data, label="action")
    plt.plot(x_datas, label="x")
    plt.plot(x_dot_datas, label="xdot")
    plt.legend(loc="best")

    plt.show()