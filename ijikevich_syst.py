import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def ijikevich_syst(a, b, c, d):
    
    def rhs(t, vec):
        v, u = vec
        
        dv_dt = v**2 + v - u
        du_dt = a*(b*v - u)
        
        return np.array([dv_dt, du_dt])

    return rhs


def calc_eq_st(b):
    v1 = 0
    u1 = 0
    v2 = b-1
    u2 = b*v2
    if v1 == v2:
        return [[v1, u1]]
    return [[v1, u1], [v2, u2]]


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


# Нахождение собственных чисел и векторов
def eig(v, a, b):
    A = np.array([[2*v + 1, -1],
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
            print(s_values)
            
    elif np.all(np.real(s_values) > 0):
        stable = False
        if np.all(np.imag(s_values) == 0):
            point_type = "Unstable Node"
        else:
            point_type = "Unstable Focus"
            print(s_values)
            
    elif np.any(np.real(s_values) > 0) and np.any(np.real(s_values) < 0):
        stable = False
        point_type = "Saddle"
        
    elif np.all(np.real(s_values) == 0) and np.all(np.imag(s_values) != 0):
        stable = True
        point_type = "Centre"
        
    elif np.any(s_values == 0) and np.any(s_values > 0):
        stable = False
        print(s_values)
        point_type = "Unstable Saddle-Node"
        
    elif np.any(s_values == 0) and np.any(s_values < 0):
        stable = True
        print(s_values)
        point_type = "Stable Saddle-Node"
    
    else:
        stable = False
        point_type = "Special point"
        print(s_values)
        
    point = [point_type, stable]
    return point


# Построение сепаратрис
def draw_separatrice(rhs, limits, v, u, a, b):
    s_values, s_vectors = eig(v, a, b)
    
    # print(f'v={v}, u={u}\ns_val={s_values}\ns_vec={s_vectors}')
    # print(s_vectors)
    
    for i, vec in enumerate(s_vectors):

        # if s_values[i] > 0:
        #     time = [0., max_time]
        # else:
        #     time = [0., -max_time]
        
        int_end = integrate_end(limits)
        time = 6

        sep1 = solve_ivp(rhs, [0., time], (v + vec[0]*0.01, u + vec[1]*0.01), method="RK45", rtol=1e-6, events=int_end)
        v1, u1 = sep1.y
        plt.plot(v1, u1, '--r')
        
        sep1 = solve_ivp(rhs, [0., -time], (v + vec[0]*0.01, u + vec[1]*0.01), method="RK45", rtol=1e-6, events=int_end)
        v1, u1 = sep1.y
        plt.plot(v1, u1, '--r')

        # sep1 = solve_ivp(rhs, [0., time], (v - vec[0]*0.01, u - vec[1]*0.01), method="RK45", rtol=1e-6, events=int_end)
        # v1, u1 = sep1.y
        # plt.plot(v1, u1, '--r')
        
        sep1 = solve_ivp(rhs, [0., -time], (v - vec[0]*0.01, u - vec[1]*0.01), method="RK45", rtol=1e-6, events=int_end)
        v1, u1 = sep1.y
        plt.plot(v1, u1, '--r')


# Построение сепаратрис
def draw_saddle_node_separatrice(rhs, limits, v, u, a, b):
    s_values, s_vectors = eig(v, a, b)
    
    # print(f'v={v}, u={u}\ns_val={s_values}\ns_vec={s_vectors}')
    # print(s_vectors)
    
    for i, vec in enumerate(s_vectors):

        # if s_values[i] > 0:
        #     time = [0., max_time]
        # else:
        #     time = [0., -max_time]
        
        int_end = integrate_end(limits)
        time = 2000

        sep1 = solve_ivp(rhs, [0., time], (v + vec[0]*0.005, u + vec[1]*0.005), method="RK45", rtol=1e-6, events=int_end)
        v1, u1 = sep1.y
        plt.plot(v1, u1, '--r')
        
        sep1 = solve_ivp(rhs, [0., -time], (v + vec[0]*0.005, u + vec[1]*0.005), method="RK45", rtol=1e-6, events=int_end)
        v1, u1 = sep1.y
        plt.plot(v1, u1, '--r')
        
        # sep1 = solve_ivp(rhs, [0., time], (v - vec[0]*0.005, u - vec[1]*0.005), method="RK45", rtol=1e-6, events=int_end)
        # v1, u1 = sep1.y
        # plt.plot(v1, u1, '--r')
        
        # sep1 = solve_ivp(rhs, [0., -time], (v - vec[0]*0.005, u - vec[1]*0.005), method="RK45", rtol=1e-6, events=int_end)
        # v1, u1 = sep1.y
        # plt.plot(v1, u1, '--r')


def integrate_end(limits):
    def out_of_limits(t, y):
        v_lims, u_lims = limits
        err = 5
        if not (v_lims[0]-err < y[0] < v_lims[1]+err and u_lims[0]-err < y[1] < u_lims[1]+err):
            # print('!!!!!!!!!!!!!', y)
            return 0
        return 1
    out_of_limits.terminal = True
    
    return out_of_limits


def calk_border_b(a):
    return (a**2 + 2*a + 1)/(4*a)


def format_num(num):
    # Форматируем число: максимум 3 знака после запятой, если они есть
    return f"{num:.3f}".rstrip('0').rstrip('.')

        
# Функция для построения портрета
def plot_plane(eq_states, ab, limits, trajectories):
    # for i, init_vals in enumerate(arr_abcd):
    a, b = ab
    c, d = 0, 0
    plt.title(f"a={format_num(a)}, b={format_num(b)}")
    print(f'\na={format_num(a)}, b={format_num(b)}')
        
    v_lims, u_lims = limits
    plt.xlim(v_lims[0], v_lims[1])
    plt.ylim(u_lims[0], u_lims[1])
        
    rhs = ijikevich_syst(a, b, c, d)
    x_vec, y_vec, U, V = eq_quiver(rhs, limits)
    plt.quiver(x_vec, y_vec, U, V, alpha=0.8)
    
    int_end = integrate_end([[-1.5, 1.5], [-1.5, 1.5]])
    time = 10

    # draw trajectories
    for traj in trajectories:
        sol = solve_ivp(rhs, [0, time], (traj[0], traj[1]), method="RK45", rtol=1e-6, events=int_end)
        v_tr, u_tr = sol.y
        plt.plot(v_tr, u_tr, 'darkgreen')
        
        sol = solve_ivp(rhs, [0, -time], (traj[0], traj[1]), method="RK45", rtol=1e-6, events=int_end)
        v_tr, u_tr = sol.y
        plt.plot(v_tr, u_tr, 'darkgreen')
    
    # draw equilibrium states
    for vu in eq_states:
        v, u = vu
        point_type = condition_type(v, a, b)
        print(format_num(v), format_num(u), point_type[0])
        
        if point_type[1] and point_type[0] != 'Stable Saddle-Node':
            plt.plot(v, u, 'bo', markersize=8)
        else:
            plt.plot(v, u, 'rx', markersize=13, markeredgewidth=2)
            
        if point_type[0] == 'Saddle':
            draw_separatrice(rhs, limits, v, u, a, b)
            
        elif point_type[0] in ('Unstable Saddle-Node', 'Stable Saddle-Node'):
            draw_saddle_node_separatrice(rhs, limits, v, u, a, b)

    plt.xlabel('V')
    plt.ylabel('U')
    plt.grid(True)
    plt.savefig(f'Phase_portrets/a={format_num(a)}_b={format_num(b)}.jpeg', format="jpeg")
    plt.show()
        
      
if __name__ == '__main__':
    # arr_ab = [[-1, -1], [-1, 1], [-1, 2]]
    # arr_eq_states = [calc_eq_st(el[1]) for el in arr_ab]
    
    # arr_limits = []
    # for eq_states in arr_eq_states:
    #     vs = [el[0] for el in eq_states]
    #     us = [el[1] for el in eq_states]
    #     arr_limits.append([[min(vs) - 0.4*(max(vs)-min(vs)) - 2, max(vs) + 0.4*(max(vs)-min(vs)) + 2],
    #                        [min(us) - 0.4*(max(us)-min(us)) - 2, max(us) + 0.4*(max(us)-min(us)) + 2]])
        
    # arr_trajectories = [[[-2, 0], [0, 2], [0, 0.5], [1, -1], [-3.5, 3]],
    #                     [[0, -1], [0, 1], [-1, 0], [1, 0]],
    #                     [[-1.5, 0], [0, -1], [0, 2], [1.5, 0], [1, 3.5], [2, 2]], 
    #                     [], [], 
    #                     [], [],
    #                     [], []]
    
    # for i, eq_states in enumerate(arr_eq_states):
    #     plot_plane(eq_states, arr_ab[i], arr_limits[i], arr_trajectories[i])
    
    ab = [0, 2]
    eq_states = calc_eq_st(ab[1])
    
    vs = [el[0] for el in eq_states]
    us = [el[1] for el in eq_states]
    limits = [[min(vs) - 0.4*(max(vs)-min(vs)) - 5, max(vs) + 0.4*(max(vs)-min(vs)) + 5],
              [min(us) - 0.4*(max(us)-min(us)) - 5, max(us) + 0.4*(max(us)-min(us)) + 5]]
    trajs = []
    
    plot_plane(eq_states, ab, limits, trajs)
