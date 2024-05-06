# SINGLE CLUSTER DESTEXE-PARÉ SIMULATION
# AUTHOR: Markéta Trembaczová, 2024

from math import sqrt
import scipy.integrate as integrate
from scipy.signal import periodogram, detrend
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx


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
    """returns adjacency matrix of a small world network of n neurons, each node is joined with its k nearest neighbors
         in a ring topology, p is the probability of rewiring each edge"""
    G = nx.watts_strogatz_graph(n, k, p, seed)
    return nx.adjacency_matrix(G).todense()


def all_to_all(n):
    """returns adjacency matrix of an all to all network of n neurons"""
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


def random_network(n, p, seed):
    """returns adjacency matrix of a random network of n neurons, p is the probability of edge creation"""
    G = nx.erdos_renyi_graph(n, p, seed, directed=False)
    return nx.adjacency_matrix(G).todense()


def ring(n):
    """returns adjacency matrix of a ring network of n neurons"""
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


######################################################################################################
# MODEL DEFINITION

def dp(p, alpha, beta):
    return alpha*(1-p) - beta*p


def destexhe_pare(t, variables, params, stimulus):
    """function for definition of the single destexhe-paré neurons, returns increments of the variables
        # variables = [Vi, mi, hi, ni, mMi]
        # params = [C, Iext, gNa, gK, VNa, VK, VT, VS, C[neuron]
        # stimulus = [st_t0[neuron], st_tn[neuron], st_A[neuron], st_r[neuron]]
        """

    # variables
    V = variables[0]
    m = variables[1]
    h = variables[2]
    n = variables[3]
    mM = variables[4]

    # parameters
    C = params[8]
    Iext = params[1]
    A = stimulus[2]
    r = stimulus[3]
    if stimulus[1] <= t <= stimulus[1] + stimulus[0]:  # adds stimulus on given interval
        Iext += A * np.exp(-r * (t - stimulus[1]))
    g_Na = params[2]
    g_Kdr = params[3]
    V_Na = params[4]
    V_K = params[5]
    V_T = params[6]
    V_S = params[7]
    g_L = 0.019
    g_M = 2
    V_L = -65

    # other variables
    alpha_m = (-0.32 * (V - V_T - 13)) / (np.exp(-(V - V_T - 13) / 4) - 1)
    beta_m = (0.28 * (V - V_T - 40)) / (np.exp((V - V_T - 40) / 5) - 1)
    dm = dp(m, alpha_m, beta_m)

    alpha_h = 0.128 * np.exp(-(V - V_T - V_S - 17) / 18)
    beta_h = 4 / (1 + np.exp(-(V - V_T - V_S - 40) / 5))
    dh = dp(h, alpha_h, beta_h)

    alpha_n = (-0.032 * (V - V_T - 15)) / (np.exp(-(V - V_T - 15) / 5) - 1)
    beta_n = 0.5 * np.exp(-(V - V_T - 10) / 40)
    dn = dp(n, alpha_n, beta_n)

    alpha_mM = (0.0001 * (V + 30)) / (1 - np.exp(-(V + 30) / 9))
    beta_mM = (-0.0001 * (V + 30)) / (1 - np.exp(V + 30) / 9)
    dmM = dp(mM, alpha_mM, beta_mM)

    # model definition
    I_L = g_L * (V - V_L)
    I_Na = g_Na * m ** 3 * h * (V - V_Na)
    I_Kdr = g_Kdr * n ** 4 * (V - V_K)
    I_M = g_M * mM * (V - V_K)

    dV = 1 / C * (Iext - I_L - I_Na - I_Kdr - I_M)

    return [dV, dm, dh, dn, dmM, C]


def coupled_DP(t, variables, K, n, params, stimulus):
    """function for adding coupling to the interneurons, returns increments of the variables
        # variables = [Vi, mi, hi, ni, mMi]
        # params = [C, Iext, gNa, gK, VNa, VK, VT, VS]
        # stimulus = [st_t0, st_tn, st_A, st_r]
        """

    Vi = variables[:n]

    dV = []
    dm = []
    dh = []
    dn = []
    dmM = []

    # computes the increment for each single neuron
    for neuron in range(n):
        stimulus_neuron = [stimulus[0]]
        for index in range(1, 4):
            stimulus_neuron.append(stimulus[index][neuron])  # correct values of stimulus for each neuron
            params[8] = params[0][neuron]  # correct value of C for each neuron
        dVar = destexhe_pare(t, [variables[neuron], variables[neuron+n], variables[neuron+2*n],  variables[neuron+3*n],
                                 variables[neuron+4*n]], params, stimulus_neuron)
        dV.append(dVar[0])
        dm.append(dVar[1])
        dh.append(dVar[2])
        dn.append(dVar[3])
        dmM.append(dVar[4])

    # adds coupling
    for row in range(n):
        for col in range(n):
            dV[row] += (Vi[col] - Vi[row]) * K[row][col] / params[0][row]

    dV.extend(dm)
    dV.extend(dh)
    dV.extend(dn)
    dV.extend(dmM)
    return dV


######################################################################################################
# EULER-MARUYAMA METODA

def euler_maruyama(sigma_noise, X0, T, dt, n_neurons, K, params, stimulus):
    """function for the Euler-Maruyama numeric method, returns solution and the time vector.
        dX =  f(X) dt + sigma dW
        sigma_noise is the standard deviation of the noised input, X0 are the initial conditions, T is the length of
        the time interval and dt is the time step, n_neurons is the number of neurons and K is the adjacency matrix of
        the network.
         params = [C, Iext, gNa, gK, VNa, VK, VT, VS]
        stimulus = [st_t0, st_tn, st_A, st_r]
    """

    N = np.floor(T / dt).astype(int)  # number of steps
    d = len(X0)  # number of differential equations
    sigma = []
    for neuron in range(n_neurons):
        sigma.append(sigma_noise / sqrt(params[0][neuron]))  # sigma_noise/sqrt(C_i)
    sigma_noise = sigma + [0] * (d - n_neurons)
    X = np.zeros((d, N + 1))
    X[:, 0] = X0
    t = np.arange(0, T + dt, dt)
    dW = np.vstack([np.sqrt(dt) * np.random.randn(n_neurons, N),  # increments of Wiener's process
                    np.zeros((d - n_neurons, N))])

    for step in range(N):
        neuron = coupled_DP(t[step], X[:, step], K, n_neurons, params, stimulus)  # solution without noise
        for i in range(len(neuron)):
            neuron[i] = neuron[i] * dt  # multiply the solution without noise by the time step
        X[:, step + 1] += X[:, step] + neuron + sigma_noise * dW[:, step]  # EM method
    return X, t


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
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n)


######################################################################################################
if __name__ == '__main__':
    # NUMBER OF NEURONS AND TYPE OF COUPLING
    n = 5
    epsilon = 0.02
    p = 0.5
    k = 3
    seed = 123

    # K = no_coupling(n)
    # coupling_type = "no coupling"
    # K = small_world(n, k, p, seed)
    # coupling_type = "small world (k=" + str(k) + ", p=" + str(p) + ")"
    K = all_to_all(n)
    coupling_type = "all to all"
    # K = random_network(n, p, seed)
    # coupling_type = "random network (p=" + str(p) + ")"
    # K = ring(n)
    # coupling_type = "ring"

    K = K * epsilon

    # INITIAL CONDITIONS
    t0 = 0
    tn = 200
    dt = 0.01

    V0_beg = 0
    V0_end = 5
    V0 = []
    m0 = []
    h0 = []
    n0 = []
    nM0 = []
    for i in range(n):
        V0.append(random.uniform(V0_beg, V0_end))
        m0.append(0.3)
        h0.append(0.42)
        n0.append(0.3)
        nM0.append(0.4)
    y0 = V0 + m0 + h0 + n0 + nM0

    # STIMULUS
    Iext = 24
    st_len = 50

    st_t0_mu = 50
    st_t0_sig = 0
    st_t0_a = 45
    st_t0_b = 55
    st_t0 = random_truncnorm(st_t0_a, st_t0_b, st_t0_mu, st_t0_sig, n)

    st_A_mu = 25
    st_A_sig = 0
    st_A_a = 20
    st_A_b = 30
    st_A = random_truncnorm(st_A_a, st_A_b, st_A_mu, st_A_sig, n)

    st_r_mu = 0.25
    st_r_sig = 0
    st_r_a = 0.1
    st_r_b = 0.3
    st_r = random_truncnorm(st_r_a, st_r_b, st_r_mu, st_r_sig, n)

    stimulus = [st_len, st_t0, st_A, st_r]

    # PARAMETERS
    C_mu = 1
    C_sigma = 0.03
    C_a = 0.91
    C_b = 1.09
    C = random_truncnorm(C_a, C_b, C_mu, C_sigma, n)
    gNa = 120
    gK = 100
    VNa = 55
    VK = -85
    VT = -58
    VS = -10
    params = [C, Iext, gNa, gK, VNa, VK, VT, VS, C[0]]

    # NOISE
    sigma_noise = 0  # standard deviation of the added noise

    ######################################################################################################
    # SOLUTION
    if sigma_noise == 0:
        # using Runge Kutta 45
        res = integrate.solve_ivp(coupled_DP, [t0, tn], y0, method='RK45', args=[K, n, params, stimulus])
        V = res.y
        T = res.t
    else:
        # using Euler Maruyama
        V, T = euler_maruyama(sigma_noise, y0, tn, dt, n, K, params, stimulus)

    ######################################################################################################
    # VISUALIZATIONS

    # COMPUTATIONS
    Vsum = []
    for i in range(len(V[0])):
        Vsum.append(0)
        for neuron in range(n):
            Vsum[i] += V[neuron][i]

    ymin = min(Vsum)
    ymax = max(Vsum)
    for neuron in range(n):
        ymin = min(ymin, min(V[neuron]))
        ymax = max(ymax, max(V[neuron]))

    text = str(n) + " neurons with " + str(coupling_type) + " coupling"

    # SINGLE NEURONS
    for neuron in range(n):
        plt.plot(T, V[neuron], 'blue')
    plt.xlabel('t')
    plt.ylabel('V')
    plt.title('Separate coupled interneurons \n' + text)
    plt.show()

    # SUM OF THE NEURONS
    plt.plot(T, Vsum, 'black')
    plt.xlabel('t')
    plt.ylabel('V')
    plt.ylim(ymin - 10, ymax + 10)
    plt.title('Sum of the coupled interneurons \n' + text)
    plt.show()

    # PERIODOGRAM
    fs = 1000 * len(Vsum) / tn
    f, Pxx = periodogram(detrend(Vsum), fs)
    kernel = np.array([1, 1, 1]) / 3  # Define the smoothing kernel
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
    plt.title('Periodogram \n' + text)
    plt.show()
