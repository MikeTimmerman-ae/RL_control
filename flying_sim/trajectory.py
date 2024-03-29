from flying_sim.drone import Drone
from flying_sim.configs.config import Config

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy.interpolate import interp1d
import matplotlib

matplotlib.use('TkAgg')


class Trajectory:
    def __init__(self, config: Config):
        self.EGO_START_POS = config.trajectory_config.EGO_START_POS
        self.EGO_FINAL_GOAL_POS = config.trajectory_config.EGO_FINAL_GOAL_POS
        self.EGO_RADIUS = config.trajectory_config.EGO_RADIUS

        self.s_0 = np.array(
            [self.EGO_START_POS[0], self.EGO_START_POS[1], 0., 0., 0., 0.])
        self.s_f = np.array([self.EGO_FINAL_GOAL_POS[0],
                            self.EGO_FINAL_GOAL_POS[1], 0., 0., 0., 0.])
        # Number of time discretization nodes (0, 1, ... N).
        self.N = config.trajectory_config.N

        self.planar_quad = Drone(config)
        # State dimension; 6 for (x, y, theta, vx, vy, omega).
        self.x_dim = self.planar_quad.x_dim
        # Control dimension; 2 for (T1, T2).
        self.u_dim = self.planar_quad.u_dim
        self.equilibrium_thrust = 0.5 * self.planar_quad.m * self.planar_quad.g

    def render_scene(self, traj=None):
        fig, ax = plt.subplots()
        ego_circle_start = plt.Circle(
            self.EGO_START_POS, radius=self.EGO_RADIUS, color='lime')
        ego_circle_end = plt.Circle(
            self.EGO_FINAL_GOAL_POS, radius=self.EGO_RADIUS, color='red')
        if traj is not None:
            for i in range(traj.shape[0]):
                x, y, theta, _, _, _ = traj[i]
                ego_circle_current = plt.Circle(
                    (x, y), radius=self.EGO_RADIUS, color='cyan')
                ax.add_patch(ego_circle_current)
                ego_arrow_current = plt.arrow(x, y, dx=np.sin(
                    theta)/2, dy=np.cos(theta)/2, head_width=0.1)
                ax.add_patch(ego_arrow_current)
        ax.add_patch(ego_circle_start)
        ax.add_patch(ego_circle_end)
        ax.set_xlim((-10.0, 10.0))
        ax.set_ylim((0.0, 10.0))
        ax.set_aspect('equal')
        return plt

    def pack_decision_variables(self, final_time: float, states: np.array, controls: np.array) -> np.array:
        """Packs decision variables (final_time, states, controls) into a 1D vector.

        Args:
            final_time: scalar.
            states: array of shape (N + 1, x_dim).
            controls: array of shape (N, u_dim).
        Returns:
            An array `z` of shape (1 + (N + 1) * x_dim + N * u_dim,).
        """
        return np.concatenate([[final_time], states.ravel(), controls.ravel()])

    def unpack_decision_variables(self, z: np.array) -> (float, np.array, np.array):
        """Unpacks a 1D vector into decision variables (final_time, states, controls).

        Args:
            z: array of shape (1 + (N + 1) * x_dim + N * u_dim,).
        Returns:
            final_time: scalar.
            states: array of shape (N + 1, x_dim).
            controls: array of shape (N, u_dim).
        """
        final_time = z[0]
        states = z[1:1 + (self.N + 1) *
                   self.x_dim].reshape(self.N + 1, self.x_dim)
        controls = z[-self.N * self.u_dim:].reshape(self.N, self.u_dim)
        return final_time, states, controls

    def optimize_trajectory(self, N=50, verbose=False) -> (float, np.array, np.array):
        equilibrium_thrust = 0.5 * self.planar_quad.m * self.planar_quad.g
        x_dim = self.planar_quad.x_dim
        u_dim = self.planar_quad.u_dim

        def cost(z):
            final_time, states, controls = self.unpack_decision_variables(z)
            dt = final_time / N
            return final_time + dt * np.sum(np.square(controls - equilibrium_thrust))

        z_guess = self.pack_decision_variables(10, self.s_0 + np.linspace(0, 1, N + 1)[:, np.newaxis] * (self.s_f - self.s_0),
                                               equilibrium_thrust * np.ones((N, u_dim)))

        bounds = Bounds(
            self.pack_decision_variables(0., -np.inf * np.ones((N + 1, x_dim)),
                                         self.planar_quad.min_thrust_per_prop * np.ones((N, u_dim))),
            self.pack_decision_variables(np.inf, np.inf * np.ones((N + 1, x_dim)),
                                         self.planar_quad.max_thrust_per_prop * np.ones((N, u_dim))))

        def equality_constraints(z):
            final_time, states, controls = self.unpack_decision_variables(z)
            dt = final_time / N
            constraint_list = [states[i + 1] - self.planar_quad.step_RK1(
                states[i], controls[i], dt) for i in range(N)]
            constraint_list.append(states[0] - self.s_0)
            constraint_list.append(states[-1] - self.s_f)
            constraint_list.append(controls[0] - self.planar_quad.init_control)
            return np.concatenate(constraint_list)

        def inequality_constraints(z):
            final_time, states, controls = self.unpack_decision_variables(z)
            # Collision avoidance
            obstacle_1 = np.sum(np.square(states[:, [0, 1]] - np.array([1, 4])), -1) - 0.5 ** 2
            obstacle_2 = np.sum(np.square(states[:, [0, 1]] - np.array([3, 6])), -1) - 0.5 ** 2
            return np.hstack((obstacle_1, obstacle_2))

        result = minimize(cost,
                          z_guess,
                          bounds=bounds,
                          constraints=[{
                              "type": "eq",
                              "fun": equality_constraints
                          }
                          ])
        if verbose:
            print(result.message)
        return self.unpack_decision_variables(result.x)

    def save_trajectory(self, save_file: str):
        tf, s, u = self.optimize_trajectory(verbose=False)
        t = np.linspace(0, tf, self.N)

        self.render_scene(s).show()

        data = np.hstack((t.reshape(self.N, 1), s[:-1], u))
        np.savetxt(save_file, data)

    def load_trajectory(self, load_file: str):
        data = np.loadtxt(load_file)
        t = data[:, 0]
        s = data[:, 1:7]
        u = data[:, 7:10]
        tf = t[-1]
        return t, s, u, tf

    def interp_trajectory(self, load_file=None):
        if load_file is None:
            tf, s, u = self.optimize_trajectory(verbose=True)
            t = np.linspace(0, tf, self.N)
            s = s[:-1]
        else:
            t, s, u, tf = self.load_trajectory(load_file)

        # self.render_scene(s)

        f_sref = interp1d(t, s, axis=0, bounds_error=False, fill_value=(s[0], s[-1]))
        f_uref = interp1d(t, u, axis=0, bounds_error=False, fill_value=(u[0], u[-1]))
        return tf, f_sref, f_uref


# trajectory = Trajectory(Config())
# trajectory.save_trajectory('trajectories/trajectory_optimal_step.csv')
