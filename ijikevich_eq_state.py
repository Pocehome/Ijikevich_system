import numpy as np

def calc_v(b):
    sq1 = 4*35**0.5
    sq2 = (b - (25 + sq1)/5) * (b - (25 - sq1)/5)
    
    return [(b-5 - sq2**0.5)/0.08, (b-5 + sq2**0.5)/0.08]


def calc_u(v):
    return 0.04*v**2 + 5*v + 140


def eig(v, a, b):
    A = np.array([[0.08*v + 5, -1],
                  [a*b, -a]])
    s_values, s_vectors = np.linalg.eig(A)
    return s_values, s_vectors


def condition_type(v, a, b):
    s_values, s_vectors = eig(v, a, b)

    if np.all(np.real(s_values) < 0):
        if np.all(np.imag(s_values) == 0):
            point_type = "Stable Node"
        else:
            point_type = "Stable Focus"
    elif np.all(np.real(s_values) > 0):
        if np.all(np.imag(s_values) == 0):
            point_type = "Unstable Node"
        else:
            point_type = "Unstable Focus"
    elif np.any(np.real(s_values) > 0) and np.any(np.real(s_values) < 0):
        point_type = "Saddle"
    elif np.all(np.real(s_values) == 0) and np.all(np.imag(s_values) != 0):
        point_type = "Centre"
    else:
        point_type = "Special point"

    return point_type


if __name__ == '__main__':
    print("D1 > 0:\n")
    for b in [0, 10]:
        v12 = calc_v(b)
        for a in [-1, 1]:
            print(f'a = {a}, b = {b}:')
            for v in v12:
                u = calc_u(v)
                point_type = condition_type(v, a, b)
                print(f'\tV = {v}, U = {u}\n\t{point_type}\n')
    
    print("\n----------------------------------------------------------------")
    print("D1 = 0:")
    for b in [(25-4*35**0.5)/5, (25+4*35**0.5)/5]:
        v = (b - 5)/0.08
        u = calc_u(v)
        for a in [-1, 1]:
            point_type = condition_type(v, a, b)
            print(f'\na = {a}, b = {b}:\n\tV = {v}, U = {u}\n\t{point_type}')
        