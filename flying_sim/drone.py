from flying_sim.configs.config import Config

import numpy as np


class Drone:
    def __init__(self, config: Config):

        self.x_dim = config.drone_config.x_dim      # state dimension
        self.u_dim = config.drone_config.u_dim      # control dimension
        self.g = config.drone_config.g              # gravity (m / s**2)
        self.m = config.drone_config.m              # mass (kg)
        self.l = config.drone_config.l              # half-length (m)
        self.I = config.drone_config.I              # moment of inertia about the out-of-plane axis (kg * m**2)
        self.Cd_v = config.drone_config.Cd_v        # translational drag coefficient
        self.Cd_phi = config.drone_config.Cd_phi    # rotational drag coefficient
        self.state_covariance = config.drone_config.state_covariance    # state update uncertainty

        # Initialize system
        self.init_state = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.init_control = np.array([0.5 * self.m * self.g, 0.5 * self.m * self.g])
        self.state = self.init_state
        self.control = self.init_control

        # Control constraints
        self.max_thrust_per_prop = 0.75 * self.m * self.g   # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0                        # until variable-pitch quadrotors become mainstream :D
        self.thrust_rate = 5 * self.m * self.g              # max change in thrust / second

    def reset(self):
        self.state = self.init_state.copy()

    def ode(self, state: np.ndarray, control: np.ndarray) -> np.array:
        """ Continuous-time dynamics of a planar quadrotor expressed as an ODE """
        assert state.shape == (
            self.x_dim,), f"State Shape: {state.shape} is not {(self.x_dim,)}"
        assert control.shape == (
            self.u_dim,), f"Control Shape: {control.shape} is not {(self.u_dim,)}"
        x, y, theta, v_x, v_y, omega = state
        T_1, T_2 = control
        return np.array([
            v_x,
            v_y,
            omega,
            (-(T_1 + T_2) * np.sin(theta) - self.Cd_v * v_x) / self.m,
            ((T_1 + T_2) * np.cos(theta) - self.Cd_v * v_y) / self.m - self.g,
            ((T_2 - T_1) * self.l - self.Cd_phi * omega) / self.I,
        ])

    def step_RK1(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.array:
        """ Discrete-time dynamics (Euler-integrated) of a planar quadrotor """
        state = state + dt * self.ode(state, control)
        return state

    def step_RK4(self, control: np.ndarray, dt: float) -> np.array:
        """ Discrete-time dynamics (Runge-Kutta 4) of a planar quadrotor """
        assert self.state.shape == (
            self.x_dim,), f"{self.state.shape} does not equal {(self.x_dim,)}"
        assert control.shape == (
            self.u_dim,), f"{control.shape} does not equal {(self.u_dim,)}"
        control = self.clip_control(control, dt)
        k1 = self.ode(self.state, control)
        k2 = self.ode(self.state + dt / 2 * k1, control)
        k3 = self.ode(self.state + dt / 2 * k2, control)
        k4 = self.ode(self.state + dt * k3, control)
        self.state += dt * (1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4) # + np.random.multivariate_normal(np.zeros(6), self.state_covariance)

    def clip_control(self, control: np.ndarray, dt: float) -> np.ndarray:
        temp = np.zeros(2)
        for i, (prev_thrust, thrust) in enumerate(zip(self.control, control)):
            if thrust > prev_thrust + self.thrust_rate * dt or thrust > self.max_thrust_per_prop:
                temp[i] = min(prev_thrust + self.thrust_rate * dt, self.max_thrust_per_prop)
            elif thrust < prev_thrust - self.thrust_rate * dt or thrust < self.min_thrust_per_prop:
                temp[i] = max(prev_thrust - self.thrust_rate * dt, self.min_thrust_per_prop)
            else:
                temp[i] = thrust
        self.control = temp
        return temp

    def get_continuous_jacobians(self, state_nominal: np.array, control_nominal: np.array) -> np.array:
        """Continuous-time Jacobians of planar quadrotor, written as a function of input state and control"""
        x, y, theta, v_x, v_y, omega = state_nominal
        T_1, T_2 = control_nominal
        A = np.array([[0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.],
                      [0., 0., -(T_1 + T_2) * np.cos(theta) /
                       self.m, -self.Cd_v / self.m, 0., 0.],
                      [0., 0., -(T_1 + T_2) * np.sin(theta) /
                       self.m, 0., -self.Cd_v / self.m, 0.],
                      [0., 0., 0., 0., 0., -self.Cd_phi / self.I]])
        B = np.array([[0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [-np.sin(theta) / self.m, -np.sin(theta) / self.m],
                      [np.cos(theta) / self.m, np.cos(theta) / self.m],
                      [-self.l / self.I, self.l / self.I]])
        return A, B
