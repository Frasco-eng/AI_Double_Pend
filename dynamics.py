import numpy as np
import matplotlib.pyplot as plt


class DoublePendulum:

    def __init__(self, m1: float = 1, m2: float = 2, l1: float = 1.5, l2: float = 1.5,
                 g: float = 9.81, b1: float = 2, b2: float = 2):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.b1 = b1
        self.b2 = b2

    def dynamics(self, states: np.ndarray,
                 inputs=None,
                 friction: bool = True,
                 complete: bool = True):
        """Compute state derivatives given current state.

        states: array-like of shape (4,) -> [o1, o1_dot, o2, o2_dot]
        inputs: optional array-like shape (2,1)
        returns: np.ndarray shape (4,) -> [o1_dot, o1_ddot, o2_dot, o2_ddot]
        """

        if inputs is None:
            inputs = np.zeros((2, 1))

        o1 = states[0]
        o1_dot = states[1]
        o2 = states[2]
        o2_dot = states[3]

        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g
        b1 = self.b1
        b2 = self.b2
        
        if inputs is None:
            inputs = np.zeros((2, 1))

        M = np.array([
            [(m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(o2), m2 * l2**2 + m2 * l1 * l2 * np.cos(o2)],
            [m2 * l2**2 + m2 *l1 *l2 * np.cos(o2), m2 * l2**2]
        ])
        
        C = np.array([
            [-m2 *l1*l2*np.sin(o2) * o2_dot, -m2 *l1*l2*np.sin(o2) * o2_dot], 
            [m2 * l1*l2*np.sin(o2) * o1_dot, m2 *l1*l2*np.sin(o2) * o1_dot]
        ])

        # C(q, q_dot) = c(q, q_dot)*q_dot
        C_complete = np.array([
            [-m2*l1*l2*np.sin(o2)*(2*o1_dot*o2_dot + o2_dot**2)],
            [m2*l1*l2*np.sin(o2)*o1_dot**2]
        ])

        if friction:
            D = np.array([
                [b1, 0],
                [0, b2]
            ])
        else:
            D = np.zeros((2, 2))

        G = np.array([
            [(m1+m2)*g*l1*np.sin(o1) +  m2*g*l2*np.sin(o1 + o2)], 
            [m2*g*l2*np.sin(o1 + o2)]
        ])

        o_dot = np.array([
            [o1_dot], 
            [o2_dot]
            ])

        if complete:
            o_dot_dot = - np.linalg.pinv(M) @ (C_complete + D @ o_dot + G + inputs)
        else:
            o_dot_dot = - np.linalg.pinv(M) @ (C @ o_dot + D @ o_dot + G + inputs)

        assert o_dot_dot.shape == (2, 1)

        o1_dot_dot = float(o_dot_dot[0, 0])
        o2_dot_dot = float(o_dot_dot[1, 0])

        states_dot = np.array([o1_dot, o1_dot_dot, o2_dot, o2_dot_dot])
        assert states_dot.shape == (4,)

        return states_dot


    def solver(self, states, states_dot, dt, method: str = 'fe'):
        if method == 'rk':
            k1 = states_dot
            k2 = self.dynamics(states=states + dt * k1 / 2)
            k3 = self.dynamics(states=states + dt * k2 / 2)
            k4 = self.dynamics(states=states + dt * k3)

            new_states = states + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif method == 'fe':
            new_states = states + dt * states_dot
        else:
            raise ValueError("choose a valid method: 'fe' or 'rk'")
        return new_states

    def animation(self, states):
        steps = states.shape[0]

        fig, ax = plt.subplots()
        plt.ion()
        plt.show()

        stop_animation = [False]  # Use list to allow modification in nested function

        def on_key(event):
            if event.key == 'q':
                stop_animation[0] = True
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)

        for i in range(steps):
            if stop_animation[0]:
                break

            ax.clear()
            o1 = states[i, 0]
            o2 = states[i, 2]

            p1 = np.array([self.l1 * np.sin(o1), -self.l1 * np.cos(o1)])
            p2 = np.array([self.l1 * np.sin(o1) + self.l2 * np.sin(o1 + o2), -self.l1 * np.cos(o1) - self.l2 * np.cos(o1 + o2)])

            lim_size = (self.l1 + self.l2) * 1.5

            ax.set_xlim([-lim_size, lim_size])
            ax.set_ylim([-lim_size, lim_size])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title(f'Double Pendulum - Step {i}')
            ax.plot([0.0, p1[0]], [0.0, p1[1]], 'grey', linewidth=2.5, label='Link 1')
            ax.scatter([0.0], [0.0], color='black', s=40, zorder=5)
            ax.scatter(p1[0], p1[1], color='blue', s=40, zorder=5)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'grey', linewidth=2.5, label='Link 2')
            ax.scatter(p2[0], p2[1], color='red', s=40, zorder=5)
            ax.legend(loc='upper right')

            plt.draw()
            plt.pause(0.001)