import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def compute_constants(p):
    """Computes commonly used constants that don't vary with the state of the
    system. Only done once initially.  Input is parameter dictionary."""

    c0 = p['I_r'] + (p['m_r'] + p['m_b']) * p['r'] ** 2
    c1 = p['I_b'] + p['m_b'] * p['l'] ** 2
    c2 = p['m_b'] * p['r'] * p['l']
    c3 = p['m_b'] * p['g'] * p['l']

    return [c0, c1, c2, c3]


def compute_variables(constants, phi):
    """Computes commonly used variables that vary with the state of the
    system, to prevent redundant computations and clean up code.  Performed
    every time step.  Input is list of constants and angle 'phi'."""

    d0 = constants[2] * np.cos(phi)
    d1 = constants[2] * np.sin(phi)
    d2 = constants[3] * np.sin(phi)
    d3 = (constants[0] + d0) / (constants[1] + d0)

    return [d0, d1, d2, d3]


def ds_dt(s, tau_func, c, p):
    """DOCSTRING"""

    # torque calculation - pass in tau_func based on controller
    tau = tau_func(s[0], s[1])

    d = compute_variables(c, s[0])

    theta_dt2 = (1 / (c[0] - d[0] * d[3])) * ((1 + d[3]) * tau +
                                              d[1] * s[1] ** 2 -
                                              p['D_v'] * s[3] +
                                              d[2] * d[3])

    phi_dt2 = (1 / (c[1] + d[0])) * (-tau + d[2] - d[0] * theta_dt2)

    x_dt2 = -(theta_dt2 + phi_dt2) * p['r']

    return [s[1], phi_dt2, s[3], theta_dt2, s[5], x_dt2]


def pd_control(x, v, Kp, Kd):
    return Kp * (-x) + Kd * (-v)


if __name__ == "__main__":

    with open('params2d.txt') as file:
        params = json.load(file)

    consts = compute_constants(params)

    t0 = 0
    tf = 15

    t_ev = np.linspace(t0, tf, 1001)

    s0 = [0.01, 0, 0, 0, 0, 0]

    sol = solve_ivp(fun=lambda t, y: ds_dt(y, lambda a, b: 0, consts, params),
                    t_span=(t0, tf),
                    y0=s0,
                    t_eval=t_ev)

    print(sol.y[0])
    print(sol.y[4])

    plt.plot(sol.t, (180 / np.pi) * sol.y[4])
    plt.show()