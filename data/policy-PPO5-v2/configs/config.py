import numpy as np
import torch


class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):

    # Configuration of Environment
    env_config = BaseConfig()
    env_config.env_name = 'flying_sim:flying_sim/PIDFlightArena-v0'
    env_config.dt = 0.01
    env_config.t0 = 0.
    env_config.seed = 50

    # Training configurations
    training = BaseConfig()
    training.num_processes = 3
    training.num_threads = 1
    training.output_dir = 'data/policy-PPO5-v2/'
    training.overwrite = True
    training.no_cuda = True  # disables CUDA training
    training.cuda = not training.no_cuda and torch.cuda.is_available()
    training.cuda_deterministic = False  # sets flags for determinism when using CUDA (potentially slow!)
    training.log_interval = 10
    training.num_env_steps = 2.4e5

    # PPO configurations
    ppo = BaseConfig()
    ppo.num_steps = 2048

    # Configuration of drone
    drone_config = BaseConfig()
    drone_config.x_dim = 6  # state dimension (see dynamics below)
    drone_config.u_dim = 2  # control dimension (see dynamics below)
    drone_config.g = 9.807  # gravity (m / s**2)
    drone_config.m = 2.5  # mass (kg)
    drone_config.l = 1.0  # half-length (m)
    # moment of inertia about the out-of-plane axis (kg * m**2)
    drone_config.I = 1.0
    drone_config.Cd_v = 0.25  # translational drag coefficient
    drone_config.Cd_phi = 0.02255  # rotational drag coefficient

    # Configuration of trajectory
    trajectory_config = BaseConfig()
    trajectory_config.EGO_START_POS, trajectory_config.EGO_FINAL_GOAL_POS = (0.0, 0.0), (9.0, 8.0)
    trajectory_config.EGO_RADIUS = 0.1
    trajectory_config.N = 50                        # Number of time discretization nodes (0, 1, ... N).
    trajectory_config.output_file = "/trajectories/trajectory_eval"
    trajectory_config.training_files = ['flying_sim/trajectories/trajectory_train_1.csv',
                                        'flying_sim/trajectories/trajectory_train_2.csv',
                                        'flying_sim/trajectories/trajectory_train_3.csv']
    trajectory_config.evaluation_files = ['flying_sim/trajectories/trajectory_eval.csv']
    trajectory_config.num_traj = len(trajectory_config.training_files)

    # Configuration of RL scheduled controller
    RL_scheduled_config = BaseConfig()
    RL_scheduled_config.lower_gains = np.array([0.6, -0.3, 4, 8, 0.6, 6])      # Kp_x, Kp_vx, Kp_th, Kp_om, Kp_y, Kp_vy
    RL_scheduled_config.upper_gains = np.array([1.5, -0.1, 7, 12, 1.5, 9])

    # Configuration of gain scheduled controller
    gain_scheduled_config = BaseConfig()
    gain_scheduled_config.Q = 1e2 * np.diag([1., 1., 0.1, 0.1, 0.1, 0.1])
    gain_scheduled_config.R = 1e1 * np.diag([1., 1.])

    # Configuration of cascaded PD controller
    cascaded_PD = BaseConfig()
    cascaded_PD.Kp_x = 1.04
    cascaded_PD.Kp_vx = -0.2
    cascaded_PD.Kp_theta = 5.6
    cascaded_PD.Kp_omega = 10
    cascaded_PD.Kp_y = 1.05
    cascaded_PD.Kp_vy = 7.5
