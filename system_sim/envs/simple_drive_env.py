import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

from system_sim.system_dynamics import SimpleDrive


class SimpleDriveEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, **kwargs):

        self.simple_drive = SimpleDrive()
        self.target = np.array([1, 1])
        self.max_step = 4

        # Environment configuration
        self.action_space = spaces.Box(low=0, high=5, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Discrete(3)

        # Criteria configuration
        self.percentage_overshoot = [10, 15, 20]

        # Log environment variables
        self.n_steps = 0

        # Configure dynamics
        self.configure(kwargs)

    def configure(self, kwargs):
        self.simple_drive.dt = kwargs['dt']
        self.simple_drive.tf = kwargs['tf']

        self.simple_drive.x_init = kwargs['x_init']
        self.simple_drive.t_init = kwargs['t_init']
        self.target = kwargs['target']

        self.simple_drive.V_max = kwargs['V_max']
        self.simple_drive.a_max = kwargs['a_max']

        print("[INFO] Finished setting up Environement")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _get_obs(self):
        obs_idx = self.observation_space.sample()
        return obs_idx

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        self.n_steps = 0

        observation = self._get_obs()
        info = self._get_info()

        # Reset environment
        super().reset(seed=seed)  # seed self.np_random
        self.simple_drive.reset()

        return observation, info

    def step(self, action):
        obs_idx = self._get_obs()
        percentage_overshoot = self.percentage_overshoot[obs_idx]

        # Retrieve trajectory from system object
        self.simple_drive.configure_gains(action)
        time, states, inputs = self.simple_drive.trajectory(self.target)
        self.simple_drive.reset()
        self.n_steps += 1

        # Set up reward
        cost = 0
        for i, (state, input) in enumerate(zip(states, inputs)):
            error = state - np.hstack((self.target, np.zeros(2, )))
            cost -= error.transpose() @ np.diag([1, 1, 0.2, 0.2]) @ error * self.simple_drive.dt
            cost -= 0.2 * np.linalg.norm(input) * self.simple_drive.dt
        overshoot = 1 if np.max(states[:, 0]) < (percentage_overshoot / 100 + 1) * self.target[0] and \
                           np.max(states[:, 1]) < (percentage_overshoot / 100 + 1) * self.target[1] else -5
        cost += overshoot

        # Reward, Observation, Info
        observation = obs_idx
        reward = cost
        terminated = self.n_steps > self.max_step
        info = self._get_info()
        info['states'] = states
        info['inputs'] = inputs
        info['time'] = time

        return observation, reward, terminated, False, info

