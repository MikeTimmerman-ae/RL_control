import numpy as np
import torch
import matplotlib

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.ppo import PPO


def main():

    # print('\n #### 1. Create Gymnasium Environment')
    # Create a wrapped, monitored VecEnv
    kwargs = {'dt': 0.01,
              'tf': 15.0,
              'x_init': np.zeros((1, 4)),
              't_init': 0,
              'target': np.array([1, 1]),
              'V_max': 5.0,
              'a_max': 2.0}
    envs = make_vec_env('system_sim:system_sim/SimpleDriveEnv-v0', 1, env_kwargs=kwargs, monitor_kwargs={})   # Create ShmemVecEnv Object (Wrapper of Vectorized Env)

    print('\n #### 1. Train RL network')
    # 1.1 create RL policy network
    model = PPO('MlpPolicy', envs,
                verbose=1,
                n_steps=128,
                tensorboard_log='data/simple_drive')

    # 1.2 train policy network
    model.learn(total_timesteps=4096,
                log_interval=1)

    # 1.3 save policy
    model.save('data/simple_drive/model')
    model.policy.save('data/simple_drive/policy')

    # 1.4 close environment
    envs.close()


if __name__ == '__main__':
    main()
