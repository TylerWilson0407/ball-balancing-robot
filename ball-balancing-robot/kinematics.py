import math
import numpy as np


class Body:

    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = inertia
        self.COM = None


class Wheel(Body):

    def __init__(self, mass, inertia, radius):
        super().__init__(mass, inertia)
        self.radius = radius


class Ball(Body):

    def __init__(self, mass, inertia, radius):
        super().__init__(mass, inertia)
        self.radius = radius


class Robot:

    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = inertia
        self.wheels = []
        self.ball = None


def rotate_3d(theta_x, theta_y, theta_z):
    pass


if __name__ == "__main__":
    ball = {'mass': None,
            'inertia': None,
            'radius': None}

    wheel = {'mass': None,
             'inertia': None,
             'radius': None}

    body = {'mass': None,
            'inertia': None}

    m_robot = 3
    I_robot = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

    # distance between robot body COM and ball COM
    com_robot = np.array([0, 0, 0.35])

    I_wheel = np.array([[0.05, 0, 0],
                        [0, 0.05, 0],
                        [0, 0, 0.05]])

    m_ball = 0.1
    I_ball = np.array([[0.1, 0, 0],
                       [0, 0.1, 0],
                       [0, 0, 0.1]])

    r_wheel = 0.05
    r_ball = 0.125

    n_wheels = 3

    # "vertical" angle of wheels on ball
    alpha_wheel = [math.radians(45)] * n_wheels

    # position and axis of wheels before rotation.
    pos_wheels = [[0, (r_ball + r_wheel), 0]] * n_wheels
    axis_wheels = [[0, 0, -1]] * n_wheels

    for i in range(n_wheels):
        pass
