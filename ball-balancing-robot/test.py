import json
import numpy as np
import scipy.linalg
import sympy as sy


def eqs_of_motion():
    """Defines and solves the equations of motion for the angular
    acceleration of the body, and the acceleration of the ball.  Returns
    expressions for them."""

    c0, c1, c2, c3 = sy.symbols('c0 c1 c2 c3')
    Dv, tau, r = sy.symbols('Dv tau r')
    phi, phi_dt, phi_dt2 = sy.symbols('phi phi_dt phi_dt2')
    xr, xr_dt, xr_dt2 = sy.symbols('xr xr_dt xr_dt2')

    # equations of motion for 2D model
    eom1 = sy.Eq(
        (2 * c0 + c2 * sy.cos(phi)) * phi_dt2 +
        (-c0 / r) * xr_dt2 +
        ((-c2 * sy.sin(phi) * phi_dt) + Dv) * phi_dt +
        (-Dv / r) * xr_dt +
        (-tau)
    )

    eom2 = sy.Eq(
        (c1 + 2 * c2 * sy.cos(phi)) * phi_dt2 +
        ((-c2 * sy.cos(phi)) / r) * xr_dt2 +
        (-c3 * sy.sin(phi)) +
        (tau)
    )

    # solve equations of motion for phi_dt2 and xr_dt2
    sol_eom = sy.solve((eom1, eom2), (phi_dt2, xr_dt2))

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


def lqr(A, B, Q, R):
    """Continuous-time LQR controller design."""

    # solve continuous algebraic Riccati equation
    X = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # derive gain matrix K
    K = np.matmul(scipy.linalg.inv(R), np.matmul(B.T, X))

    # return eigenvalues
    eig, __ = scipy.linalg.eig(A - np.matmul(B, K))

    return K, X, eig


if __name__ == "__main__":

    param_file = 'params2d.txt'

    # import parameters
    with open(param_file) as file:
        params = json.load(file)

    # solve ODE
    consts = compute_constants(params)

    c0, c1, c2, c3 = sy.symbols('c0 c1 c2 c3')
    Dv, tau, r = sy.symbols('Dv tau r')
    phi, phi_dt = sy.symbols('phi phi_dt')
    xr, xr_dt = sy.symbols('xr xr_dt')

    phi_dt2, xr_dt2 = eqs_of_motion()

    A = sy.Matrix([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

    B = sy.Matrix([0, 0, 0, 0])

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

    A = np.float_(A.subs(setpoint))
    B = np.float_(B.subs(setpoint))

    # Q matrix
    Q = np.zeros((4, 4))

    # max errors for state
    e_phi = np.radians(10)  # 10 degrees
    e_xr = 0.5  # 0.5m
    e_phidt = 3 * e_phi  # derivatives estimated 3x position
    e_xrdt = 3 * e_xr

    for i, e in enumerate([e_phi, e_xr, e_phidt, e_xrdt]):
        Q[i][i] = 1 / e ** 2

    # R matrix

    # max torque - 3.75Nm
    u_max = 3.75

    R = [[(1 / u_max ** 2)]]

    K, X, eig = lqr(A, B, Q, R)

