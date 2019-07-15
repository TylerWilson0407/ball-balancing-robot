import json

# all units in metric (kg, m)

if __name__ == "__main__":

    params2d = {

        'g': 9.81,          # gravitational constant

        'D_v': 5,        # viscous damping constant

        'm_b': 7.135,       # body mass
        'm_r': 3.2,         # ball mass

        'l': 0.405,         # distance between ball/body centers of mass
        'r': 0.115,       # radius of ball

        'I_b': 2.4,         # body inertia about rotation axis
        'I_r': 2.65e-2     # ball inertia
    }

    with open('params2d.txt', 'w') as file:
        json.dump(params2d, file)
