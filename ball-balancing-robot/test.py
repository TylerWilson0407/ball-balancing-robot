import json
import numpy as np
import scipy.linalg
import sympy as sy


def sym_dict2d():

    sd = {}

    # constants c0 - c3 (functions of mass, length, inertia, gravity constant)
    sd['c0'], sd['c1'], sd['c2'], sd['c3'] = sy.symbols('c0 c1 c2 c3')
    sd['Dv'], sd['r'] = sy.symbols('Dv r')
    sd['tau'] = sy.symbols('tau')
    sd['phi'], sd['phi_dt'], sd['phi_dt2'] = sy.symbols('phi phi_dt phi_dt2')
    sd['x'], sd['x_dt'], sd['x_dt2'] = sy.symbols('x x_dt x_dt2')
    
    return sd


def solve_eom_2d():
    """Solve equations of motion for two acceleration variables phi_dt2 and
    x_dt2 for the 2D model.

    phi = angle of robot body
    x = horizontal position of ball

    Returns Sympy expressions

    """
    
    s = sym_dict2d()

    eom1 = sy.Eq(
        (2 * s['c0'] + s['c2'] * sy.cos(s['phi'])) * s['phi_dt2'] +
        (-s['c0'] / s['r']) * s['x_dt2'] +
        ((-s['c2'] * sy.sin(s['phi']) * s['phi_dt']) + s['Dv']) * s['phi_dt'] +
        (-s['Dv'] / s['r']) * s['x_dt'] +
        (-s['tau'])
    )

    eom2 = sy.Eq(
        (s['c1'] + 2 * s['c2'] * sy.cos(s['phi'])) * s['phi_dt2'] +
        ((-s['c2'] * sy.cos(s['phi'])) / s['r']) * s['x_dt2'] +
        (-s['c3'] * sy.sin(s['phi'])) +
        (s['tau'])
    )

    sol = sy.solve((eom1, eom2), (s['phi_dt2'], s['x_dt2']))

    return [sol[s['phi_dt2']], sol[s['x_dt2']]]


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

    # import defined parameters
    param_file = 'params2d.txt'
    params = json.load(open(param_file))

    # solve ODE
    consts = compute_constants(params)

    # define symbols used in equations of motion
    # c0, c1, c2, c3 = sy.symbols('c0 c1 c2 c3')
    # Dv, tau, r = sy.symbols('Dv tau r')
    # phi, phi_dt, phi_dt2 = sy.symbols('phi phi_dt phi_dt2')
    # x, x_dt, x_dt2 = sy.symbols('x x_dt x_dt2')

    # eom1 = sy.Eq(
    #     (2 * c0 + c2 * sy.cos(phi)) * phi_dt2 +
    #     (-c0 / r) * x_dt2 +
    #     ((-c2 * sy.sin(phi) * phi_dt) + Dv) * phi_dt +
    #     (-Dv / r) * x_dt +
    #     (-tau)
    # )
    #
    # eom2 = sy.Eq(
    #     (c1 + 2 * c2 * sy.cos(phi)) * phi_dt2 +
    #     ((-c2 * sy.cos(phi)) / r) * x_dt2 +
    #     (-c3 * sy.sin(phi)) +
    #     (tau)
    # )

    s = sym_dict2d()

    [phi_dt2, x_dt2] = solve_eom_2d

    A = sy.Matrix([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

    B = sy.Matrix([0, 0, 0, 0])

    for i, s_dt in enumerate([phi_dt2, x_dt2]):

        for j, s in enumerate([s['phi'], s['x'], s['phi_dt'], s['x_dt']]):
            A[i + 2, j] = s_dt.diff(s)

        B[i + 2] = s_dt.diff(s['tau'])

    setpoint = {s['phi']: 0,
                s['x']: 0,
                s['phi_dt']: 0,
                s['x_dt']: 0,
                s['r']: params['r'],
                s['Dv']: params['D_v'],
                s['c0']: consts[0],
                s['c1']: consts[1],
                s['c2']: consts[2],
                s['c3']: consts[3]}

    A = np.float_(A.subs(setpoint))
    B = np.float_(B.subs(setpoint))

    # Q matrix
    Q = np.zeros((4, 4))

    # max errors for state
    e_phi = np.radians(10)  # 10 degrees
    e_x = 0.5  # 0.5m
    e_phidt = 3 * e_phi  # derivatives estimated 3x position
    e_xdt = 3 * e_x

    for i, e in enumerate([e_phi, e_x, e_phidt, e_xdt]):
        Q[i][i] = 1 / e ** 2

    # R matrix

    # max torque - 3.75Nm
    u_max = 3.75

    R = [[(1 / u_max ** 2)]]

    K, X, eig = lqr(A, B, Q, R)

    print(K)