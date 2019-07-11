import json
import numpy as np
from sympy import *


def eqs_of_motion():
    """Solves for the angular accelerations of the body and ball.  Returns
    expressions for them."""

    c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    Dv = symbols('Dv')
    tau = symbols('tau')
    r = symbols('r')
    phi, theta = symbols('phi theta')
    phi_dt, phi_dt2, theta_dt, theta_dt2 = symbols('phi_dt phi_dt2 theta_dt '
                                                   'theta_dt2')
    xr, xr_dt, xr_dt2 = symbols('xr xr_dt xr_dt2')

    # equations of motion for 2D model
    eq1 = Eq(
        (c0 * theta_dt2) + (c0 + (c2 * cos(phi))) * phi_dt2 - (c2 * sin(phi))
        * phi_dt ** 2 - tau + Dv * theta_dt)

    eq2 = Eq((c2 * cos(phi) * theta_dt2) +
             ((c1 + (c2 * cos(phi))) * phi_dt2) -
             (c3 * sin(phi)) + tau)

    theta_dt2_sub = phi_dt2 - (1 / r) * xr_dt2
    theta_dt_sub = phi_dt - (1 / r) * xr_dt

    x_sub = {theta_dt2: theta_dt2_sub,
             theta_dt: theta_dt_sub}

    pprint(eq1.subs(x_sub))
    pprint(eq2.subs(x_sub))

    sol_eq = solve((eq1, eq2), (theta_dt2, phi_dt2))

    theta_dt2_expr = sol_eq[theta_dt2].subs(phi_dt2,
                                            sol_eq[phi_dt2] - phi_dt2)
    phi_dt2_expr = sol_eq[phi_dt2].subs(theta_dt2,
                                        sol_eq[theta_dt2] - theta_dt2)

    return theta_dt2_expr, phi_dt2_expr


def compute_constants(p):
    """Computes commonly used constants that don't vary with the state of the
    system. Only done once initially.  Input is parameter dictionary."""

    c0 = p['I_r'] + (p['m_r'] + p['m_b']) * p['r'] ** 2
    c1 = p['I_b'] + p['m_b'] * p['l'] ** 2
    c2 = p['m_b'] * p['r'] * p['l']
    c3 = p['m_b'] * p['g'] * p['l']

    return [c0, c1, c2, c3]


if __name__ == "__main__":

    param_file = 'params2d.txt'

    # import parameters
    with open(param_file) as file:
        params = json.load(file)

    # solve ODE
    consts = compute_constants(params)

    c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    # Dv = symbols('Dv')
    tau = symbols('tau')
    phi, theta, xr = symbols('phi theta xr')
    phi_dt, theta_dt, xr_dt = symbols('phi_dt theta_dt xr_dt')

    theta_dt2, phi_dt2 = eqs_of_motion()

    theta_dt2_dphi = theta_dt2.diff(phi)
    theta_dt2_dtau = theta_dt2.diff(tau)
    theta_dt2_dxr = theta_dt2.diff(xr)
    phi_dt2_dphi = phi_dt2.diff(phi)
    phi_dt2_dtau = phi_dt2.diff(tau)
    phi_dt2_dxr = phi_dt2.diff(xr)

    setpoint = {phi: 0,
                theta: 0,
                phi_dt: 0,
                theta_dt: 0,
                xr: 0,
                xr_dt: 0,
                c0: consts[0],
                c1: consts[1],
                c2: consts[2],
                c3: consts[3]}

    # state of system X = [phi, xr, phi_dt, xr_dt]

    A = Matrix([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, theta_dt2_dphi, 0, 0],
                [0, phi_dt2_dphi, 0, 0]])

    B = Matrix([0, 0, theta_dt2_dtau, phi_dt2_dtau])

    A = np.float_(A.subs(setpoint)).tolist()
    B = np.float_(B.subs(setpoint)).tolist()

    # print(A)
    # print(B)

    params['A'] = A
    params['B'] = B

    # print(params)
    # print(json.dumps(A))

    params['A'] = A
    params['B'] = B

    with open(param_file, 'w') as file:
        json.dump(params, file)