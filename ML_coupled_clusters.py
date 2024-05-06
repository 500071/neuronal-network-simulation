# MULTIPLE CLUSTER MORRIS-LECAR SIMULATION
# AUTHOR: Markéta Trembaczová, 2024

import numpy as np
from math import sqrt
import random
import networkx as nx
from scipy.stats import truncnorm
import scipy.integrate as integrate
from scipy.signal import periodogram, detrend
import plotly.graph_objects as go
import matplotlib.pyplot as plt


######################################################################################################
# NETWORKS FOR COUPLING

def no_coupling(n):
    """returns adjacency matrix of a network of n neurons with no coupling"""
    K = []
    for row in range(n):
        line = []
        for col in range(n):
            line.append(0)
        K.append(line)
    return np.asarray(K)


def small_world(n, k, p, seed):
    """ function for INNER COUPLING
    returns adjacency matrix of a small world network of n neurons, each node is joined with its k nearest neighbors
    in a ring topology, p is the probability of rewiring each edge"""
    G = nx.watts_strogatz_graph(n, k, p, seed)
    return nx.adjacency_matrix(G).todense()


def small_world_outer(n, k, p, seed):
    """ function for OUTER COUPLING
    returns adjacency matrix of a small world network of n neurons, each node is joined with its k nearest neighbors
    in a ring topology, p is the probability of rewiring each edge"""
    np.random.seed(seed)
    # regural structure
    K = np.zeros((n, n))
    for i in range(k):
        K = K + np.diag([1] * (n - i), i) + np.diag([1] * i, i - n)
    # rewiring
    K_random = np.random.random((n, n))
    K_index = np.random.randint(0, n * n, size=(n, n))
    for row in range(n):
        for col in range(n):
            if K[row][col] == 1:
                if K_random[row][col] < p:
                    K[row][col] = 0
                    index = K_index[row][col]
                    K[index // n][index % n] = 1
    return np.asarray(K)


def all_to_all(n):
    """function for INNER COUPLING
    returns adjacency matrix of an all to all network of n neurons"""
    K = []
    for row in range(n):
        line = []
        for col in range(n):
            if row == col:
                line.append(0)
            else:
                line.append(1)
        K.append(line)
    return np.asarray(K)


def all_to_all_outer(n):
    """function for OUTER COUPLING
        returns adjacency matrix of an all to all network of n neurons"""
    return np.ones((n, n))


def random_network(n, p, seed):
    """function for INNER COUPLING
    returns adjacency matrix of a random network of n neurons, p is the probability of edge creation"""
    G = nx.erdos_renyi_graph(n, p, seed, directed=False)
    return nx.adjacency_matrix(G).todense()


def random_network_outer(n, p, seed):
    """function for OUTER COUPLING
        returns adjacency matrix of a random network of n neurons, p is the probability of edge creation"""
    np.random.seed(seed)
    K = np.random.random((n, n))
    for row in range(n):
        for col in range(n):
            if K[row][col] < p:
                K[row][col] = 1
            else:
                K[row][col] = 0
    return np.asarray(K)


def ring(n):
    """function for INNER COUPLING
    returns adjacency matrix of a ring network of n neurons"""
    if n == 1:
        return np.asarray([[0]])
    K = []
    for row in range(n):
        line = []
        for col in range(n):
            if col == (row-1) % n or col == (row+1) % n:
                line.append(1)
            else:
                line.append(0)
        K.append(line)
    return np.asarray(K)


def ring_outer(n):
    """function for OUTER COUPLING
        returns adjacency matrix of a ring network of n neurons"""
    K = np.zeros((n, n))
    K[n-1][0] = 1
    return np.asarray(K)


######################################################################################################
# MODEL DEFINITION

def morris_lecar(t, variables, params, stimulus):
    """function for definition of the single morris-lecar neuron, returns increments of the variables [dVi, dhi, dni]
            # variables = [Vi, wi]
            # params = [C[neuron], Iext, gK, gCa, gL, VK, VCa, VL]
            # stimulus = [st_t0[neuron], st_tn[neuron], st_A[neuron], st_r[neuron]]
            """

    # variables
    Vi = variables[0]
    wi = variables[1]

    # parameters
    C = params[0]
    Iext = params[1]
    A = stimulus[2]
    r = stimulus[3]
    if stimulus[1] <= t <= stimulus[1] + stimulus[0]:  # adds stimulus on given interval
        Iext += A * np.exp(-r * (t - stimulus[1]))
    gK = params[2]
    gCa = params[3]
    gL = params[4]
    VK = params[5]
    VCa = params[6]
    VL = params[7]

    beta1 = -1.2
    beta2 = 18
    beta3 = 5
    beta4 = 17.4
    phi = 1 / 15

    # model definition
    minf = (1 + np.tanh((Vi - beta1) / beta2)) / 2
    winf = (1 + np.tanh((Vi - beta3) / beta4)) / 2
    tauinf = (1 / (phi * np.cosh((Vi - beta3) / (2 * beta4)))) / 2

    dVi = (-gCa * minf * (Vi - VCa) - gK * wi * (Vi - VK) - gL * (Vi - VL) + Iext) / C
    dwi = (winf - wi) / tauinf

    return [dVi, dwi]


def coupledML(t, variables, K, n_neurons, n_clusters, params, stimulus):
    """function for adding coupling to the morris-lecar neurons, returns increments of the variables [dVi, wi]
        # variables = [Vi, wi]
        # params = [C, Iext, gK, gCa, gL, VK, VCa, VL], all the parameters are lists with values for each cluster
        # stimulus = [st_t0, st_tn, st_A, st_r], all the parameters are lists with values for each cluster
        """

    Vi = variables[:n_neurons*n_clusters]

    dV = []
    dw = []

    # computes the increment for each single neuron
    for cluster in range(n_clusters):
        for neuron in range(n_neurons):
            params_neuron = [params[0][cluster][neuron]]  # correct value of C for each neuron
            for index in range(1, 8):  # correct values of parameters for each neuron
                params_neuron.append(params[index][cluster])
            stimulus_neuron = [stimulus[0][cluster]]  # correct length of stimulus for each neuron
            for index in range(1, 4):
                stimulus_neuron.append(stimulus[index][cluster][neuron])  # correct values of stimulus for each neuron
            dVar = morris_lecar(t, [variables[neuron + cluster * n_neurons],
                                    variables[neuron + n_neurons * (n_clusters + cluster)]],
                                params_neuron, stimulus_neuron)
            dV.append(dVar[0])
            dw.append(dVar[1])

    # adds coupling
    for row in range(n_neurons * n_clusters):
        for col in range(n_neurons * n_clusters):
            dV[row] += (Vi[col] - Vi[row]) * K[row][col] / params[0][row % n_clusters][row // n_clusters]

    dV.extend(dw)
    return dV


######################################################################################################
# EULER-MARUYAMA METODA

def euler_maruyama(sigma_noise, X0, T, dt, n_neurons, n_clusters, K, params, stimulus):
    """function for the Euler-Maruyama numeric method, returns solution and the time vector.
        dX =  f(X) dt + sigma dW
        sigma_noise is the standard deviation of the noised input, X0 are the initial conditions, T is the length of
        the time interval and dt is the time step, n_clusters is the number of clusters, n_neurons is the number of
        neurons in each cluster and K is the adjacency matrix of the network.
        params = [C, Iext, gNa, gK, VNa, VK]
        stimulus = [st_t0, st_tn, st_A, st_r]
    """

    N_eq = n_neurons*n_clusters  # number of all the neurons in the system
    N = np.floor(T/dt).astype(int)  # number of steps
    d = len(X0)  # number of differential equations
    sigma = []
    for cluster in range(n_clusters):
        for neuron in range(n_neurons):
            sigma.append(sigma_noise / sqrt(params[0][cluster][neuron]))  # sigma_noise/sqrt(C_i)
    sigma_noise = sigma + [0] * (d - N_eq)
    X = np.zeros((d, N + 1))
    X[:, 0] = X0
    t = np.arange(0, T+dt, dt)
    dW = np.vstack([np.sqrt(dt) * np.random.randn(N_eq, N),  # increments of Wiener's process
                    np.zeros((d - N_eq, N))])

    for step in range(N):
        neuron = coupledML(t[step], X[:, step], K, n_neurons, n_clusters, params, stimulus)
        for i in range(len(neuron)):
            neuron[i] = neuron[i] * dt  # multiply the solution without noise by the time step
        X[:, step + 1] += X[:, step] + neuron + sigma_noise * dW[:, step]  # EM method
    return X, t


######################################################################################################
# GENERATE K FUNCTIONS

def generate_K(coupling_type, n, k, p, seed):
    """function for generating the inner coupling matrices
        1: no coupling, 2: all to all, 3: smallworld, 4: random network, 5: ring
    """
    if coupling_type == 2:
        K = all_to_all(n)
    elif coupling_type == 3:
        K = small_world(n, k, p, seed)
    elif coupling_type == 4:
        K = random_network(n, p, seed)
    elif coupling_type == 5:
        K = ring(n)
    else:
        K = no_coupling(n)
    return K


def generate_outer_K(coupling_type, n, k, p, seed, loc):
    """function for generating the outer coupling matrices
            1: no coupling, 2: all to all, 3: smallworld, 4: random network, 5: ring
        """
    if coupling_type == 2:
        K = all_to_all_outer(n)
    elif coupling_type == 3:
        K = small_world_outer(n, k, p, seed)
    elif coupling_type == 4:
        K = random_network_outer(n, p, seed)
    elif coupling_type == 5:
        K = ring_outer(n)
    else:
        K = no_coupling(n)

    if loc == "U":
        return K
    else:
        return np.transpose(K)


######################################################################################################
# TRUNCATED NORMAL DISTRIBUTION


def random_truncnorm(a, b, mu, sigma, n):
    """returns n numbers from truncated normal distribution TN(mu, sigma, a, b)"""
    if sigma == 0:
        if n == 1:
            return [mu]
        else:
            return [mu] * n
    a = (a - mu) / sigma
    b = (b + mu) / sigma
    return list(truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n))


######################################################################################################
if __name__ == '__main__':
    # NUMBER OF CLUSTERS AND NEURONS
    n_clusters = 2
    n_neurons = 5

    # TYPE AND STRENGTH OF THE COUPLING
    # diagonal is the inner coupling, rest outer coupling of clusters
    # 1: no coupling, 2: all to all, 3: smallworld, 4: random network, 5: ring
    epsilon = [[0.001, 0.0002], [0.0002, 0.001]]
    coupling = [[2, 2], [2, 2]]
    k = [[0, 0], [0, 0]]
    p = [[0, 0], [0, 0]]
    seed = [[123, 123], [123, 123]]

    # GENERATE COUPLING MATRIX K
    K_list = []
    for row in range(n_clusters):
        line = []
        for col in range(n_clusters):
            if row == col:  # diagonal
                K = epsilon[row][col] * \
                    generate_K(coupling[row][col], n_neurons, k[row][col], p[row][col], int(seed[row][col]))
            elif row < col:  # upper triangle
                K = epsilon[row][col] * \
                    generate_outer_K(coupling[row][col], n_neurons, k[row][col], p[row][col], int(seed[row][col]),
                                     "U")
            else:  # lower triangle
                K = epsilon[row][col] * \
                    generate_outer_K(coupling[row][col], n_neurons, k[row][col], p[row][col], int(seed[row][col]),
                                     "L")
            line.append(K)
        K_list.append(line)
    K = np.concatenate([np.concatenate(row, axis=1) for row in K_list], axis=0)

    # NETWORK VISUALIZATION (optional)
    fig_K = go.Figure(data=go.Heatmap(
        z=K,
        colorscale="purples",
        colorbar={"title": 'epsilon'}))
    fig_K.layout.height = 500
    fig_K.layout.width = 500
    fig_K.update_yaxes(autorange="reversed")
    fig_K.update_layout(
        title='Visualisation of the coupling matrix')
    # fig_K.show()

    # INITIAL CONDITIONS
    V0_beg = [35, -35]
    V0_end = [35, -35]
    w0_input = [0.5, 0.5]

    V0 = []
    w0 = []
    for cluster in range(n_clusters):
        for neuron in range(n_neurons):
            V0.append(random.uniform(V0_beg[cluster], V0_end[cluster]))
            w0.append(w0_input[cluster])
    y0 = V0 + w0

    # TIME
    t0 = 0
    tn = 200
    dt = 0.01

    # STIMULUS
    Iext = [100, 100]
    st_len = [20, 20]

    st_t0_mu = [0, 0]
    st_t0_sig = [0, 0]
    st_t0_a = [0, 0]
    st_t0_b = [0, 0]
    st_t0 = []
    for cluster in range(n_clusters):
        st_t0.append(
            random_truncnorm(st_t0_a[cluster], st_t0_b[cluster], st_t0_mu[cluster], st_t0_sig[cluster], n_neurons))

    st_A_mu = [0, 0]
    st_A_sig = [0, 0]
    st_A_a = [0, 0]
    st_A_b = [0, 0]
    st_A = []
    for cluster in range(n_clusters):
        st_A.append(
            random_truncnorm(st_A_a[cluster], st_A_b[cluster], st_A_mu[cluster], st_A_sig[cluster], n_neurons))

    st_r_mu = [0, 0]
    st_r_sig = [0, 0]
    st_r_a = [0, 0]
    st_r_b = [0, 0]
    st_r = []
    for cluster in range(n_clusters):
        st_r.append(
            random_truncnorm(st_r_a[cluster], st_r_b[cluster], st_r_mu[cluster], st_r_sig[cluster], n_neurons))

    stimulus = [st_len, st_t0, st_A, st_r]

    # PARAMETERS
    C_mu = [1, 1]
    C_sigma = [0.001, 0.001]
    C_a = [0.91, 0.91]
    C_b = [1.09, 1.09]
    C = []
    for cluster in range(n_clusters):
        C.append(
            random_truncnorm(C_a[cluster], C_b[cluster], C_mu[cluster], C_sigma[cluster], n_neurons))

    gK = [8, 8]
    gCa = [4, 4]
    gL = [2, 2]
    VK = [-80, -80]
    VCa = [120, 120]
    VL = [-60, -60]
    params = [C, Iext, gK, gCa, gL, VK, VCa, VL]

    # NOISE
    sigma_noise = 0  # standard deviation of the added noise

    ######################################################################################################
    # SOLUTION
    if sigma_noise == 0:
        # using Runge Kutta 45
        res = integrate.solve_ivp(coupledML, [t0, tn], y0, method='RK45',
                                  args=[K, n_neurons, n_clusters, params, stimulus])
        V = res.y
        T = res.t
    else:
        # using Euler Maruyama
        V, T = euler_maruyama(sigma_noise, y0, tn, dt, n_neurons, n_clusters, K, params, stimulus)

    ######################################################################################################
    # VISUALIZATIONS

    # COMPUTATIONS
    Vsum = []
    for i in range(len(V[0])):
        Vsum.append(0)
        for neuron in range(n_neurons*n_clusters):
            Vsum[i] += V[neuron][i]

    ymin = min(Vsum)
    ymax = max(Vsum)
    for neuron in range(n_neurons*n_clusters):
        ymin = min(ymin, min(V[neuron]))
        ymax = max(ymax, max(V[neuron]))

    # SINGLE NEURONS
    colors = ["blue", "red", "green"]
    for neuron in range(n_neurons*n_clusters):
        plt.plot(T, V[neuron], colors[neuron % n_clusters])
    plt.xlabel('t')
    plt.ylabel('V')
    plt.show()

    # SUM OF THE NEURONS
    plt.plot(T, Vsum, 'black')
    plt.xlabel('t')
    plt.ylabel('V')
    plt.ylim(ymin - 10, ymax + 10)
    plt.show()

    # PERIODOGRAM
    fs = 1000 * len(Vsum) / tn
    f, Pxx = periodogram(detrend(Vsum), fs)
    kernel = np.array([1, 1, 1]) / 3
    smoothed_Pxx = np.convolve(Pxx, kernel, mode='same')
    f_smoothed = f[:len(smoothed_Pxx)]

    limit = 2000
    threshold = 0.1

    for i in range(round(limit / 2), len(f_smoothed)):
        if smoothed_Pxx[i] > threshold:
            limit = i * 5 + 100

    plt.plot(f_smoothed, smoothed_Pxx)
    plt.xlim(0, limit)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('|P(f)|')
    plt.show()
