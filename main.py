from system_sim.system_dynamics import SimpleDrive

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from stable_baselines3.ppo.ppo import PPO

matplotlib.use('TkAgg')


# Instantiate RL policy and Simple Drive model
RL_control = PPO.load('data/simple_drive/model')
simple_drive = SimpleDrive()

# Configure Simple Drive
percentage_overshoot_ls = [10, 15, 20]
obs_idx = 2
percentage_overshoot = percentage_overshoot_ls[obs_idx]
gains = RL_control.predict(np.array(obs_idx), deterministic=True)[0]
print(gains)
simple_drive.configure_gains(gains)

# Simulate Dynamics
time, states, inputs = simple_drive.trajectory(np.array([1, 1]))

# Plot results
fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(time, states[:, 0])
axs[0, 0].plot(time, (percentage_overshoot / 100 + 1) * np.ones(len(time)), '-r')
axs[0, 0].grid()
axs[0, 0].set_ylabel('x-position [m]')

axs[0, 1].plot(time, states[:, 1], label='Trajectory')
axs[0, 1].plot(time, (percentage_overshoot / 100 + 1) * np.ones(len(time)), '-r', label='Max. Overshoot')
axs[0, 1].grid()
axs[0, 1].set_ylabel('y-position [m]')
axs[0, 1].legend(loc='lower right')

axs[1, 0].plot(time, states[:, 2])
axs[1, 0].grid()
axs[1, 0].set_ylabel('x-velocity [m/s]')

axs[1, 1].plot(time, states[:, 3])
axs[1, 1].grid()
axs[1, 1].set_ylabel('y-velocity [m/s]')

axs[2, 0].plot(time, inputs[:, 0])
axs[2, 0].grid()
axs[2, 0].set_xlabel('Time [s]')
axs[2, 0].set_ylabel('x-acceleration [m/s2]')

axs[2, 1].plot(time, inputs[:, 1])
axs[2, 1].grid()
axs[2, 1].set_xlabel('Time [s]')
axs[2, 1].set_ylabel('y-acceleration [m/s2]')

plt.show()

