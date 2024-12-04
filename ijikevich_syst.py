import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def ijikevich_syst(a, b, c, d):
    
    def rhs(t, vec):
        v, u = vec
        
        # if v >= 30:
        #     return np.array([-c - v, u + d])
        # if v >= 30:
        #     v = c
        #     u += d
        
        dv_dt = 0.04*v**2 + 5*v + 140 - u
        du_dt = a*(b*v - u)
        
        return np.array([dv_dt, du_dt])

    return rhs


# Функция для построения векторного поля
def eq_quiver(rhs, limits, N=16):
    x_lims, y_lims = limits
    x_vec = np.linspace(x_lims[0], x_lims[1], N)
    y_vec = np.linspace(y_lims[0], y_lims[1], N)
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    for i, y in enumerate(y_vec):
        for j, x in enumerate(x_vec):
            vfield = rhs(0., [x, y])
            u, v = vfield
            U[i][j] = u
            V[i][j] = v
    return x_vec, y_vec, U, V


def calc_v(a, b):
    coefficients = [a * 0.04, a * (5 - b), a * 140]
    roots = np.roots(coefficients)
    
    if roots[0] == roots[1] or 0 < roots[0].imag < 1.e-5:
        roots = np.array([roots[0].real])
    
    return roots


def calc_u(v):
    return 0.04*v**2 + 5*v + 140


# Нахождение собственных чисел и векторов
def eig(v, a, b):
    A = np.array([[0.08*v + 5, -1],
                  [a*b, -a]])
    s_values, s_vectors = np.linalg.eig(A)
    return s_values, s_vectors


# Определение состояния равновесия
def condition_type(v, a, b):
    s_values, s_vectors = eig(v, a, b)

    if np.all(np.real(s_values) < 0):
        stable = True
        if np.all(np.imag(s_values) == 0):
            point_type = "Stable Node"
        else:
            point_type = "Stable Focus"
            
    elif np.all(np.real(s_values) > 0):
        stable = False
        if np.all(np.imag(s_values) == 0):
            point_type = "Unstable Node"
        else:
            point_type = "Unstable Focus"
            
    elif np.any(np.real(s_values) > 0) and np.any(np.real(s_values) < 0):
        stable = False
        point_type = "Saddle"
        
    elif np.all(np.real(s_values) == 0) and np.all(np.imag(s_values) != 0):
        stable = True
        point_type = "Centre"
        
    else:
        stable = False
        point_type = "Special point"
        
    point = [point_type, stable]
    return point


# Построение сепаратрис
def draw_separatrice(rhs, limits, v, u, a, b):
    s_values, s_vectors = eig(v, a, b)
    
    # print(f'v={v}, u={u}\ns_val={s_values}\ns_vec={s_vectors}')
    
    for i, vec in enumerate(s_vectors):

        # if v == 0 and i == 1:
        #     continue

        if s_values[i] < 0:
            time = [0., 8.]
        else:
            time = [0., -10.]

        sep1 = solve_ivp(rhs, time, (v + vec[0]*0.01, u + vec[1]*0.01), method="RK45", rtol=1e-12)
        v1, u1 = sep1.y
        plt.plot(v1, u1, '--r')

        sep1 = solve_ivp(rhs, time, (v - vec[0]*0.01, u - vec[1]*0.01), method="RK45", rtol=1e-12)
        v1, u1 = sep1.y
        plt.plot(v1, u1, '--r')


def format_num(num):
    # Форматируем число: максимум 3 знака после запятой, если они есть
    return f"{num:.3f}".rstrip('0').rstrip('.')

        
# Функция для построения портрета
def plot_plane(arr_abcd, arr_limits, arr_traj):
    for i, init_vals in enumerate(arr_abcd):
        a, b, c, d = init_vals
        plt.title(f"a={format_num(a)}, b={format_num(b)}")
        
        limits = arr_limits[i]
        v_lims, u_lims = limits
        plt.xlim(v_lims[0], v_lims[1])
        plt.ylim(u_lims[0], u_lims[1])
        
        rhs = ijikevich_syst(a, b, c, d)
        x_vec, y_vec, U, V = eq_quiver(rhs, limits)
        plt.quiver(x_vec, y_vec, U, V, alpha=0.8)
        
        for traj in arr_traj[i]:
            sol = solve_ivp(rhs, [0, 16], (traj[0], traj[1]), method="RK45", rtol=1e-6)
            v_tr, u_tr = sol.y
            plt.plot(v_tr, u_tr, 'darkgreen')
        
        v12 = calc_v(a, b)
        for v in v12:
            point_type = condition_type(v, a, b)
            u = calc_u(v)
            
            if point_type[1]:
                plt.plot(v, u, 'bo', markersize=8)
            else:
                plt.plot(v, u, 'rx', markersize=13, markeredgewidth=2)
                
            if point_type[0] == "Saddle":
                draw_separatrice(rhs, limits, v, u, a, b)
                
            print(v, u, point_type[0])
        print()
    
        plt.xlabel('V')
        plt.ylabel('U')
        plt.grid(True)
        plt.show()
        
      
if __name__ == '__main__':
    
    # arr_eq_st = [[[-1, 0.00000, -82.65564, 0.00000], [-1, 0.00000, -42.34436, 0.00000]],
    #              [[1, 0.00000, -82.65564, 0.00000], [1, 0.00000, -42.34436, 0.00000]],
    #              [[-1, 10.00000, 42.34436, 423.44356], [-1, 10.00000, 82.65564, 826.55644]],
    #              [[1, 10.00000, 42.34436, 423.44356], [1, 10.00000, 82.65564, 826.55644]],
    #              [[-1, 0.26714, -59.16080, -15.80399]],
    #              [[1, 0.26714, -59.16080, -15.80399]],
    #              [[-1, 9.73286, 59.16080, 575.80399]],
    #              [[1, 9.73286, 59.16080, 575.80399]]]
    
    c = -20.
    d = 10.
    b_border = [(25.-4*35**0.5)/5, (25.+4*35**0.5)/5]
    
    arr_abcd = [[-1., 0., c, d], [1., 0., c, d],
                [-1., 10., c, d], [1., 10., c, d],
                [-1, b_border[0], c, d], [1, b_border[0], c, d],
                [-1, b_border[1], c, d], [1, b_border[1], c, d]]
                # [-1, b_border[0] + 1, c, d], [1, b_border[0] + 1, c, d],
                # [-1, b_border[1] - 1, c, d], [1, b_border[1] - 1, c, d]]
                
    arr_limits = [[(-100, -20), (-10, 10)], [(-100, -20), (-10, 10)],
                  [(20, 105), (300, 1000)], [(20, 105), (300, 1000)],
                  [(-70, -45), (-25, -10)], [(-70, -45), (-25, -10)],
                  [(50, 70), (550, 600)], [(50, 70), (550, 600)]]
    
    arr_traj = [[(-100, 1), (-100, -1), (-42.15, 0.1), (-42.33, -0.1), (-42.35, 0.01), (-43, -0.1)],
                [(-100, 10), (-100, -10), (-45, 10), (-50, -10), (-35, 10), (-45, -10)],
                [],
                [],
                [(-70, -18), (-70, -19), (-55, -14)],
                [(-68, -10), (-58, -10), (-70, -25), (-53, -10)],
                [(60, 580), (57, 570)],
                [(55, 580), (60, 550)]]
    
    plot_plane(arr_abcd, arr_limits, arr_traj)