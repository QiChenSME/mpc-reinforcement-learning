from numba.core.cgutils import terminate

from project.PIDController import PIDController, InputType, FeedbackMethod
from CustomEnv.customtest import MassBlockEnv
from CustomEnv.CartpoleProMax import CartPoleV3, CartPoleVectorTest
import matplotlib.pyplot as plt

if __name__ == "__main__":
    steps = 500
    act_data = []
    x_datas = []
    x_dot_datas = []
    theta_datas = []
    theta_dot_datas = []

    # env = MassBlockEnv()
    env = CartPoleVectorTest(render_mode='human')
    env.reset()

    kp = 16
    kd = 4
    controller = PIDController(kp=kp, kd=kd, input_type=InputType.DUAL, feedback_method=FeedbackMethod.Negative)
    controller.reset(env.state[2])

    terminated = False
    for i in range(steps):
        if not terminated:
            action = controller.pid(0, env.state[2], env.state[3])

            data, _, _, _, _ = env.step(action)

            act_data.append(action)
            x_datas.append(data[0])
            x_dot_datas.append(data[1])
            theta_datas.append(data[2])
            theta_dot_datas.append(data[3])
        else:
            env.close()

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

    plt.show()