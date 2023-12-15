import numpy as np


class SimpleDrive:
    def __init__(self):
        self.dt = 0.01
        self.x_init = np.zeros((1, 4))
        self.t_init = 0
        self.tf = 20

        self.state = self.x_init
        self.time = self.t_init

        # Controller Configuration
        self.Kp_x = 2
        self.Kp_y = 2
        self.Kp_Vx = 1
        self.Kp_Vy = 1

        # System Configuration
        self.V_max = 5
        self.a_max = 2

    def reset(self):
        self.state = self.x_init
        self.time = self.t_init

    def configure_gains(self, action):
        self.Kp_x = action[0]
        self.Kp_y = action[1]
        self.Kp_Vx = action[2]
        self.Kp_Vy = action[3]

    def controller(self, reference):
        vx_ref = self.Kp_x * (reference[0] - self.state[0, 0])
        vy_ref = self.Kp_y * (reference[1] - self.state[0, 1])

        vx_ref = np.clip(vx_ref, -self.V_max, self.V_max)
        vy_ref = np.clip(vy_ref, -self.V_max, self.V_max)

        ax = self.Kp_Vx * (vx_ref - self.state[0, 2])
        ay = self.Kp_Vy * (vy_ref - self.state[0, 3])

        ax = np.clip(ax, -self.a_max, self.a_max)
        ay = np.clip(ay, -self.a_max, self.a_max)
        return np.array([ax, ay])

    def EOM(self, control):
        x_dot = np.zeros((1, 4))

        x_dot[0, 0] = self.state[0, 2]
        x_dot[0, 1] = self.state[0, 3]
        x_dot[0, 2] = control[0]
        x_dot[0, 3] = control[1]

        x_next = self.state + self.dt * x_dot
        return x_next

    def trajectory(self, target: np.ndarray):
        state_ls = self.state
        input_ls = np.zeros((1, 2))
        time_ls = [self.time]
        target_reached = True

        while target_reached and self.time < self.tf:
            control_input = self.controller(target)
            self.state = self.EOM(control_input).reshape(1, 4)
            self.time += self.dt

            target_reached = np.linalg.norm(self.state[0, :2] - target) > 1e-2 or np.linalg.norm(self.state[0, 2:]) > 1e-2

            state_ls = np.vstack((state_ls, self.state.copy()))
            input_ls = np.vstack((input_ls, control_input.copy()))
            time_ls.append(self.time)
        return time_ls, state_ls, input_ls
