import logging
from typing import Any, Optional
from matplotlib import pyplot as plt
from gymnasium.wrappers import TimeLimit

from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent, LstdDpgAgent, UpdateStrategy
from mpcrl.optim import NetwonMethod, GradientDescent
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes
from mpcrl.core.exploration import *

from Env import StageEnv, Tracer
from Dynamics import WaferStage
from Agent import *
from MPC import *

if __name__ == '__main__':
    render_mode = None
    episode_steps = 500
    episodes = 10

    model = WaferStage()
    env = MonitorEpisodes(Tracer(StageEnv(model, render_mode=render_mode, debug_mode=True,
                                          w_x=[1000, 1000, 1000, 5000, 5000, 5000, 50, 50, 50, 100, 100, 100]),
                                 max_episode_steps=episode_steps,
                                 # mode="dynamic",
                                 # init_values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 # trans_list={100: [0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 #             300: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
                                 )
                          )
    mpc = NonLinearMpc()
    learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
            for name, val in mpc.learnable_pars_init.items()
        )
    )
    # noinspection PyTypeChecker
    agent = Log(
        RecordUpdates(
            NonLinearLstdQLearningAgent(
                mpc=mpc,
                learnable_parameters=learnable_pars,
                fixed_parameters=mpc.fixed_pars_init,
                discount_factor=mpc.discount_factor,
                update_strategy=1,
                optimizer=NetwonMethod(learning_rate=1e-2),
                hessian_type="approx",
                record_td_errors=True,
                remove_bounds_on_initial_action=True,
                use_last_action_on_fail=True,
                # exploration=EpsilonGreedyExploration(0.01, 1, hook="on_timestep_end"),
            )
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 10},
    )
    agent.train(env=env, episodes=episodes, seed=69, raises=False)
