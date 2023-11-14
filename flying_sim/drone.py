from control import NonlinearIOSystem
import numpy as np


class Drone:
    def __init__(self, config):
        self.dt = config.dt
        self.tf = config.tf
        self.m = config.m
        self.I = config.I
        self.l = config.l
        self.thrust_mag = config.thrust_mag

        self.t = config.t0
        self.x = config.x0
        self.solver = self.dynamics()

    def dynamics(self) -> NonlinearIOSystem:
        """ Returns instance of 'NonlinearIOSsytem' """

        # Define Update Equation
        def updfcn(t, x, u, params):
            T1, T2 = u  # Thrust values
            px, py, theta = x[:3]  # Configuration variables
            vx, vy, omega = x[3:6]  # Velocity variables

            # Calculate next state
            x_next = np.zeros(6)

            # Configuration variable update
            x_next[0] = px + self.dt * vx
            x_next[1] = py + self.dt * vy
            x_next[2] = theta + self.dt * omega

            # Configuration variable update
            x_next[3] = vx + self.dt * \
                (T1 + T2) * params.thrust_mag * np.sin(theta) / params.m
            x_next[4] = vy + self.dt * \
                (T1 + T2) * params.thrust_mag * np.cos(theta) / params.m - 9.81
            x_next[5] = omega + self.dt * \
                (T2 - T1) * params.thrust_mag * params.l / params.I

            return x_next

        # Define Output Equation
        def outfcn(t, x, u, params):
            return x

        # Create Sovler
        solver = NonlinearIOSystem(updfcn, outfcn)
        return solver

    def __str__(self) -> str:
        return "Drone"

    def step(self, u, params):
        self.x = self.solver.dynamics(self.t, self.x, u, params)
        self.t += self.dt