import json
import numpy as np
from sympy import *


def eqs_of_motion():
    """Defines and solves the equations of motion for the angular
    acceleration of the body, and the acceleration of the ball.  Returns
    expressions for them."""

    c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    Dv, tau, r = symbols('Dv tau r')
    phi, phi_dt, phi_dt2 = symbols('phi phi_dt phi_dt2')
    xr, xr_dt, xr_dt2 = symbols('xr xr_dt xr_dt2')

    # equations of motion for 2D model
    eom1 = Eq(
        (2 * c0 + c2 * cos(phi)) * phi_dt2 +
        (-c0 / r) * xr_dt2 +
        ((-c2 * sin(phi) * phi_dt) + Dv) * phi_dt +
        (-Dv / r) * xr_dt +
        (-tau)
    )

    eom2 = Eq(
        (c1 + 2 * c2 * cos(phi)) * phi_dt2 +
        ((-c2 * cos(phi)) / r) * xr_dt2 +
        (-c3 * sin(phi)) +
        (tau)
    )

    # solve equations of motion for phi_dt2 and xr_dt2
    sol_eom = solve((eom1, eom2), (phi_dt2, xr_dt2))

    # remove xr_dt2 terms from phi_dt2 expression and visa versa
    phi_dt2_equals = sol_eom[phi_dt2].subs(xr_dt2, sol_eom[xr_dt2] - xr_dt2)
    xr_dt2_equals = sol_eom[xr_dt2].subs(phi_dt2, sol_eom[phi_dt2] - phi_dt2)

    return phi_dt2_equals, xr_dt2_equals


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
    Dv, tau, r = symbols('Dv tau r')
    phi, phi_dt = symbols('phi phi_dt')
    xr, xr_dt = symbols('xr xr_dt')

    phi_dt2, xr_dt2 = eqs_of_motion()

    A = Matrix([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

    B = Matrix([0, 0, 0, 0])

    for i, s_dt in enumerate([phi_dt2, xr_dt2]):

        for j, s in enumerate([phi, xr, phi_dt, xr_dt]):
            A[i + 2, j] = s_dt.diff(s)

        B[i + 2] = s_dt.diff(tau)

    setpoint = {phi: 0,
                xr: 0,
                phi_dt: 0,
                xr_dt: 0,
                r: params['r'],
                Dv: params['D_v'],
                c0: consts[0],
                c1: consts[1],
                c2: consts[2],
                c3: consts[3]}

    pprint(A.subs(setpoint))
    pprint(B.subs(setpoint))

    A = np.float_(A.subs(setpoint)).tolist()
    B = np.float_(B.subs(setpoint)).tolist()

    params['A'] = A
    params['B'] = B

    with open(param_file, 'w') as file:
        json.dump(params, file)
