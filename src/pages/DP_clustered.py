# PAGE FOR MULTIPLE CLUSTER DESTEXHE-PARÉ SIMULATION
# AUTHOR: Markéta Trembaczová, 2024

from dash import html, register_page, dcc
from dash import Input, Output, callback, State
from math import sqrt
import scipy.integrate as integrate
from scipy.stats import truncnorm
from scipy.signal import periodogram, detrend
import numpy as np
import random
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import re


register_page(
    __name__,
    name='Destexhe-Paré clustered',
    top_nav=True,
    path='/DP_clustered'
)


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
    # regular structure
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

def dp(p, alpha, beta):
    return alpha*(1-p) - beta*p


def destexhe_pare(t, variables, params, stimulus):
    """function for definition of the single destexhe-paré neurons, returns increments of the variables
        # variables = [Vi, mi, hi, ni, mMi]
        # params = [C[neuron], Iext, gNa, gK, VNa, VK, VT, VS]
        # stimulus = [st_t0[neuron], st_tn[neuron], st_A[neuron], st_r[neuron]]
        """

    # variables
    V = variables[0]
    m = variables[1]
    h = variables[2]
    n = variables[3]
    mM = variables[4]

    # parameters
    C = params[0]
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


def coupled_DP(t, variables, K, n_neurons, n_clusters, params, stimulus):
    """function for adding coupling to the interneurons, returns increments of the variables
        # variables = [Vi, mi, hi, ni, mMi]
        # params = [C, Iext, gNa, gK, VNa, VK, VT, VS], all the parameters are lists with values for each cluster
        # stimulus = [st_t0, st_tn, st_A, st_r], all the parameters are lists with values for each cluster
        """

    Vi = variables[:n_neurons*n_clusters]

    dV = []
    dm = []
    dh = []
    dn = []
    dmM = []

    # computes the increment for each single neuron
    for cluster in range(n_clusters):
        for neuron in range(n_neurons):
            params_neuron = [params[0][cluster][neuron]]  # correct value of C for each neuron
            for index in range(1, 8):
                params_neuron.append(params[index][cluster])  # correct values of parameters for each neuron
            stimulus_neuron = [stimulus[0][cluster]]  # correct length of stimulus for each neuron
            for index in range(1, 4):
                stimulus_neuron.append(stimulus[index][cluster][neuron])  # correct values of stimulus for each neuron
            dVar = destexhe_pare(t, [variables[neuron + cluster * n_neurons],
                                     variables[neuron + n_neurons * (n_clusters + cluster)],
                                     variables[neuron + n_neurons * (2 * n_clusters + cluster)],
                                     variables[neuron + n_neurons * (3 * n_clusters + cluster)],
                                     variables[neuron + n_neurons * (4 * n_clusters + cluster)]],
                                 params_neuron, stimulus_neuron)
            dV.append(dVar[0])
            dm.append(dVar[1])
            dh.append(dVar[2])
            dn.append(dVar[3])
            dmM.append(dVar[4])

    # adds coupling
    for row in range(n_neurons * n_clusters):
        for col in range(n_neurons * n_clusters):
            dV[row] += (Vi[col] - Vi[row]) * K[row][col] / params[0][row % n_clusters][row // n_clusters]

    dV.extend(dm)
    dV.extend(dh)
    dV.extend(dn)
    dV.extend(dmM)
    return dV


######################################################################################################
# EULER-MARUYAMA METODA

def euler_maruyama(sigma_noise, X0, T, dt, n_neurons, n_clusters, K, params, stimulus):
    """function for the Euler-Maruyama numeric method, returns solution and the time vector.
            dX =  f(X) dt + sigma dW
            sigma_noise is the standard deviation of the noised input, X0 are the initial conditions, T is the length of
            the time interval and dt is the time step, n_clusters is the number of clusters, n_neurons is the number of
            neurons in each cluster and K is the adjacency matrix of the network.
            params = [C, Iext, gNa, gK, VNa, VK, VT, VS]
            stimulus = [st_t0, st_tn, st_A, st_r]
        """

    N_eq = n_neurons * n_clusters  # number of all the neurons in the system
    N = np.floor(T / dt).astype(int)  # number of steps
    d = len(X0)  # number of differential equations
    sigma = []
    for cluster in range(n_clusters):
        for neuron in range(n_neurons):
            sigma.append(sigma_noise / sqrt(params[0][cluster][neuron]))  # sigma_noise/sqrt(C_i)
    sigma_noise = sigma + [0] * (d - N_eq)
    X = np.zeros((d, N + 1))
    X[:, 0] = X0
    t = np.arange(0, T + dt, dt)
    dW = np.vstack([np.sqrt(dt) * np.random.randn(N_eq, N),  # increments of Wiener's process
                    np.zeros((d - N_eq, N))])

    for step in range(N):
        neuron = coupled_DP(t[step], X[:, step], K, n_neurons, n_clusters, params, stimulus)
        for i in range(len(neuron)):
            neuron[i] = neuron[i] * dt  # multiply the solution without noise by the time step
        X[:, step + 1] += X[:, step] + neuron + sigma_noise * dW[:, step]  # EM method
    return X, t


######################################################################################################
# PARSERS

def input_parser(input_string, n, integer=False):
    """parser for sting inputs of the variables
       input: 'a; b; c', output: [a, b, c]"""
    res = input_string.replace(',', '.').split(";")
    if len(res) != n:
        return "input error"
    if " " in res:
        return "input error"
    if integer:
        for i in range(n):
            res[i] = int(res[i])
        return res
    for i in range(n):
        res[i] = float(res[i])
    return np.asarray(res)


def matrix_input_parser(input_string, n_clusters, integer=False):
    """parser for sting inputs of the variables in the form of matrices
           input: '[a;b]; [c; d]', output: [[a,b]; [c, d]]"""
    input_string = input_string.replace(',', '.')
    input_string = input_string.replace('X', '0')
    pattern = r'\[([^]]+)\]'
    matches = re.findall(pattern, input_string)
    result = []
    for match in matches:
        if integer:
            numbers = [int(num.strip()) for num in match.split(';')]
        else:
            numbers = [float(num.strip()) for num in match.split(';')]
        result.append(numbers)
    if len(result) != n_clusters:
        return "input error"
    return np.asarray(result)


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
# LAYOUT

def layout():
    layout = html.Div([
        html.Br(),
        html.H1("Destexhe-Paré simulation — multiple clusters"),
        "Enter values for individual clusters separated by semicolons. Enter matrices in the form of [a;b];[c;d].",
        html.Br(),
        ######################################################################
        html.H5("Number of neurons and type of coupling"),
        html.Div([
            "n_clusters = ", dcc.Input(id='n_clusters', value=2.0, type='number', size='8'), " ",
            "n_neurons = ", dcc.Input(id='n_neurons', value=3.0, type='number', size='8'), " ",
        ], title="number of clusters and number of neurons in each cluster"),
        html.Br(),
        html.H6("coupling inside the clusters"),
        "1: no coupling, 2: all to all, 3: smallworld, 4: random network, 5: ring",
        html.Div([
            html.Div([
                " epsilon = ", dcc.Input(id='eps_clusters', value="0.001; 0.001", type='text', size='15'), " ",
            ], title="coupling strength"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " coupling = ", dcc.Input(id='coupling_clusters', value="2; 2", type='text', size='15'),
            ], title="coupling type"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " p = ", dcc.Input(id='p_clusters', type='text', size='15', value="0.5; 0.5"),
            ], title="random networks: probability of edge creation \n"
                     "small world networks: probability of rewiring the edges"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " k = ", dcc.Input(id='k_clusters', type='text', size='15', value="2;2"),
            ], title="small world networks: initial number of neighbours of each node before the rewiring the edges"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " seed = ", dcc.Input(id='seed_clusters', type='text', size='15', value="123;123"),
            ], title="seed for random and small world networks"),
        ], style=dict(display='flex')),
        html.Br(),
        ######################################################################
        html.H6("Outer coupling of the clusters"),
        html.Div([
            html.Div([
                " epsilon = ", dcc.Input(id='eps_outer', value="[X;0.0002]; [0.0002;X]", type='text', size='40'), " ",
            ], title="coupling strength"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " coupling = ", dcc.Input(id='coupling_outer', value="[X;2]; [2;X]", type='text', size='20'),
            ], title="coupling type"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " p = ", dcc.Input(id='p_outer', value="[X;0]; [0;X]", type='text', size="20"),
            ], title="random networks: probability of edge creation \n"
                     "small world networks: probability of rewiring the edges"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " k = ", dcc.Input(id='k_outer', value="[X;0]; [0;X]", type='text', size='20'),
            ], title="small world networks: initial number of neighbours of each node before the rewiring the edges"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " seed = ", dcc.Input(id='seed_outer', value="[X;123]; [123;X]", type='text', size='20'),
            ], title="seed for random and small world networks"),
        ], style=dict(display='flex')),
        html.Br(),

        ######################################################################
        html.H5("Initial conditions"),
        html.Div([
            html.Div([
                "V0 ∈ ( ", dcc.Input(id='V0beg', value="5; 5", type='text', size='15'),
                ", ", dcc.Input(id='V0end', value="5; 5", type='text', size='15'), " )",
            ], title="initial conditions for V0, random for each neuron from uniform distribution"),
            html.Span(' ', style={'display': 'inline-block', 'width': '30px'}),
            html.Div([
                " m0 = ", dcc.Input(id='m0', value="0.3;0.3", type='text', size='8'),
                " h0 = ", dcc.Input(id='h0', value="0.42;0.42", type='text', size='8'),
                " n0 = ", dcc.Input(id='n0', value="0.3;0.3", type='text', size='8'),
                " mM0 = ", dcc.Input(id='mM0', value="0.4;0.4", type='text', size='8'),
            ], title="initial conditions for the other variables, fixed for all neurons in the cluster"),
            html.Span(' ', style={'display': 'inline-block', 'width': '30px'}),
            html.Div([
                " t0 = ( ", dcc.Input(id='t0', value=0, type='number', size='5'),
                ", ", dcc.Input(id='tn', value=200, type='number', size='4'), " )",
                " dt = ", dcc.Input(id='dt', value=0.01, type='number', size='5'),
            ], title="time interval and step")
        ], style=dict(display='flex')),
        html.Br(),

        ######################################################################
        html.H5("Parameters"),
        html.Div([
            html.Div([
                "I_ext = ", dcc.Input(id='Iext', value="43; 43", type='text', size='15'),
            ], title="external DC current"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "sigma_noise = ", dcc.Input(id='sigma_noise', value=0, type='number', size='5'),
            ], title="standard deviation of the noise of the external current"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "C ∼ TN(",
                "μ = ", dcc.Input(id='C_mu', value="1; 1", type='text', size='15'),
                " σ = ", dcc.Input(id='C_sigma', value="0.003; 0.003", type='text', size='15'),
                " [ ", dcc.Input(id='C_a', value="0.91; 0.91", type='text', size='15'),
                ", ", dcc.Input(id='C_b', value="1.09; 1.09", type='text', size='15'), " ])",
            ], title="membrane capacitance, random from truncated normal distribution with given parameters for"
                     " each neuron"),
        ], style=dict(display='flex')),
        html.Br(),
        html.Div([
            "g_Na = ", dcc.Input(id='gNa', value="120; 120", type='text', size='15'),
            " g_K = ", dcc.Input(id='gK', value="100; 100", type='text', size='15'),
            " V_Na = ", dcc.Input(id='VNa', value="55; 55", type='text', size='15'),
            " V_K = ", dcc.Input(id='VK', value="-85; -85", type='text', size='15'),
            " V_T = ", dcc.Input(id='VT', value="-58;-58", type='text', size='15'),
            " V_S = ", dcc.Input(id='VS', value="-10;-10", type='text', size='15'),
        ], title="Destexhe-Paré parameters"),
        html.Br(),

        ######################################################################
        html.H5("STIMULUS"),
        html.Div([
            html.Div([
                "t0_st ∼ TN(",
                "μ = ", dcc.Input(id='st_t0', value="0; 0", type='text', size='15'),
                " σ = ", dcc.Input(id='st_t0_sigma', value="0; 0", type='text', size='15'),
                " [ ", dcc.Input(id='st_t0_a', value="0; 0", type='text', size='15'),
                ", ", dcc.Input(id='st_t0_b', value="0; 0", type='text', size='15'), " ])",
            ], title="start of the stimulus, random from truncated normal distribution with given parameters for "
                     "each neuron"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "T_st = ", dcc.Input(id='st_tn', value="0; 0", type='text', size='15')
            ], title="length of the stimulus"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
        ], style=dict(display='flex')),
        html.Br(),
        html.Div([
            html.Div([
                "A ∼ TN(",
                "μ = ", dcc.Input(id='st_A', value="0; 0", type='text', size='10'),
                " σ = ", dcc.Input(id='st_A_sigma', value="0; 0", type='text', size='10'),
                " [ ", dcc.Input(id='st_A_a', value="0; 0", type='text', size='10'),
                ", ", dcc.Input(id='st_A_b', value="0; 0", type='text', size='10'), " ])",
            ], title="amplitude of the stimulus, random from truncated normal distribution with given parameters "
                     "for each neuron"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "r ∼ TN(",
                "μ = ", dcc.Input(id='st_r', value="0; 0", type='text', size='10'),
                " σ = ", dcc.Input(id='st_r_sigma', value="0; 0", type='text', size='10'),
                " [ ", dcc.Input(id='st_r_a', value="0; 0", type='text', size='10'),
                ", ", dcc.Input(id='st_r_b', value="0; 0", type='text', size='10'), " ])",
            ], title=" damping rate of the stimulus, random from truncated normal distribution with given parameters "
                     "for each neuron"),
        ], style=dict(display='flex')),
        html.Br(),

        ######################################################################
        html.H5("Periodogram window"),
        html.Div([
            "from = ", dcc.Input(id='periodogram_a', value=0, type='number', size='5'),
            " to = ", dcc.Input(id='periodogram_b', value=200, type='number', size='5'),
        ], title="window for the computation of the periodogram, default is the whole interval"),
        html.Br(),

        ######################################################################
        html.Button(id='button_DPC', n_clicks=0, children="Redraw graph"),
        html.Div([], id='plots_DPC'),

    ], style={'margin-left': '110px'})
    return layout


######################################################################################################
# CALLBACK

@callback(
    Output(component_id='plots_DPC', component_property='children'),
    Input('button_DPC', 'n_clicks'),
    [State('n_clusters', 'value'), State('n_neurons', 'value'), State('eps_clusters', 'value'),
     State('coupling_clusters', 'value'), State('p_clusters', 'value'), State('k_clusters', 'value'),
     State('seed_clusters', 'value'), State('eps_outer', 'value'), State('coupling_outer', 'value'),
     State('p_outer', 'value'), State('k_outer', 'value'), State('seed_outer', 'value'), State('V0beg', 'value'),
     State('V0end', 'value'), State('m0', 'value'), State('h0', 'value'), State('n0', 'value'), State('mM0', 'value'),
     State('t0', 'value'), State('tn', 'value'), State('dt', 'value'), State('Iext', 'value'), State('C_mu', 'value'),
     State('C_sigma', 'value'), State('C_a', 'value'), State('C_b', 'value'), State('gNa', 'value'),
     State('gK', 'value'), State('VNa', 'value'), State('VK', 'value'), State('VT', 'value'), State('VS', 'value'),
     State('st_t0', 'value'), State('st_t0_sigma', 'value'), State('st_t0_a', 'value'), State('st_t0_b', 'value'),
     State('st_tn', 'value'), State('st_A', 'value'), State('st_A_sigma', 'value'), State('st_A_a', 'value'),
     State('st_A_b', 'value'), State('st_r', 'value'), State('st_r_sigma', 'value'), State('st_r_a', 'value'),
     State('st_r_b', 'value'), State('sigma_noise', 'value'), State('periodogram_a', 'value'),
     State('periodogram_b', 'value')
     ]
)
def update_output(n_clicks, n_clusters, n_neurons, epsilon_clusters, coupling_clusters, p_clusters, k_clusters,
                  seed_clusters, epsilon_outer, coupling_outer, p_outer, k_outer, seed_outer, V0_beg, V0_end, m0, h0,
                  n0, mM0, t0, tn, dt, Iext, C_mu, C_sigma, C_a, C_b, gNa, gK, VNa, VK, VT, VS, st_t0_mu, st_t0_sig,
                  st_t0_a, st_t0_b, st_len, st_A_mu, st_A_sig, st_A_a, st_A_b, st_r_mu, st_r_sig, st_r_a, st_r_b,
                  sigma_noise, periodogram_a, periodogram_b):
    if n_clicks > 0:
        # COUPLING PARAMETERS - DIAGONAL
        epsilon_clusters = input_parser(epsilon_clusters, n_clusters)
        coupling_clusters = input_parser(coupling_clusters, n_clusters, integer=True)
        k_clusters = input_parser(k_clusters, n_clusters, integer=True)
        p_clusters = input_parser(p_clusters, n_clusters)
        seed_clusters = input_parser(seed_clusters, n_clusters, integer=True)

        # COUPLING PARAMETERS - OTHER
        epsilon = matrix_input_parser(epsilon_outer, n_clusters)
        coupling = matrix_input_parser(coupling_outer, n_clusters, integer=True)
        k = matrix_input_parser(k_outer, n_clusters, integer=True)
        p = matrix_input_parser(p_outer, n_clusters)
        seed = matrix_input_parser(seed_outer, n_clusters, integer=True)

        # COUPLING PARAMETERS
        np.fill_diagonal(epsilon, epsilon_clusters)
        np.fill_diagonal(coupling, coupling_clusters)
        np.fill_diagonal(k, k_clusters)
        np.fill_diagonal(p, p_clusters)
        np.fill_diagonal(seed, seed_clusters)

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

        # INITIAL CONDITIONS
        V0_beg = input_parser(V0_beg, n_clusters)
        V0_end = input_parser(V0_end, n_clusters)
        m0_input = input_parser(m0, n_clusters)
        h0_input = input_parser(h0, n_clusters)
        n0_input = input_parser(n0, n_clusters)
        mM0_input = input_parser(mM0, n_clusters)
        V0 = []
        m0 = []
        h0 = []
        n0 = []
        mM0 = []
        for cluster in range(n_clusters):
            for neuron in range(n_neurons):
                V0.append(random.uniform(V0_beg[cluster], V0_end[cluster]))
                m0.append(m0_input[cluster])
                h0.append(h0_input[cluster])
                n0.append(n0_input[cluster])
                mM0.append(mM0_input[cluster])
        y0 = V0 + m0 + h0 + n0 + mM0

        # PARAMETERS
        Iext = input_parser(Iext, n_clusters)
        C_mu = input_parser(C_mu, n_clusters)
        C_sigma = input_parser(C_sigma, n_clusters)
        C_a = input_parser(C_a, n_clusters)
        C_b = input_parser(C_b, n_clusters)
        C = []
        for cluster in range(n_clusters):
            C.append(
                random_truncnorm(C_a[cluster], C_b[cluster], C_mu[cluster], C_sigma[cluster], n_neurons))

        gNa = input_parser(gNa, n_clusters)
        gK = input_parser(gK, n_clusters)
        VNa = input_parser(VNa, n_clusters)
        VK = input_parser(VK, n_clusters)
        VT = input_parser(VT, n_clusters)
        VS = input_parser(VS, n_clusters)
        params = [C, Iext, gNa, gK, VNa, VK, VT, VS]

        # STIMULUS
        st_len = input_parser(st_len, n_clusters)

        st_t0_mu = input_parser(st_t0_mu, n_clusters)
        st_t0_sig = input_parser(st_t0_sig, n_clusters)
        st_t0_a = input_parser(st_t0_a, n_clusters)
        st_t0_b = input_parser(st_t0_b, n_clusters)
        st_t0 = []
        for cluster in range(n_clusters):
            st_t0.append(
                random_truncnorm(st_t0_a[cluster], st_t0_b[cluster], st_t0_mu[cluster], st_t0_sig[cluster], n_neurons))

        st_A_mu = input_parser(st_A_mu, n_clusters)
        st_A_sig = input_parser(st_A_sig, n_clusters)
        st_A_a = input_parser(st_A_a, n_clusters)
        st_A_b = input_parser(st_A_b, n_clusters)
        st_A = []
        for cluster in range(n_clusters):
            st_A.append(
                random_truncnorm(st_A_a[cluster], st_A_b[cluster], st_A_mu[cluster], st_A_sig[cluster], n_neurons))

        st_r_mu = input_parser(st_r_mu, n_clusters)
        st_r_sig = input_parser(st_r_sig, n_clusters)
        st_r_a = input_parser(st_r_a, n_clusters)
        st_r_b = input_parser(st_r_b, n_clusters)
        st_r = []
        for cluster in range(n_clusters):
            st_r.append(
                random_truncnorm(st_r_a[cluster], st_r_b[cluster], st_r_mu[cluster], st_r_sig[cluster], n_neurons))

        stimulus = [st_len, st_t0, st_A, st_r]

        ######################################################################################################
        # SOLUTION
        if sigma_noise == 0:
            # using Runge Kutta 45
            res = integrate.solve_ivp(coupled_DP, [t0, tn], y0, method='RK45',
                                      args=[K, n_neurons, n_clusters, params, stimulus])
            V = res.y
            T = res.t
        else:
            # using Euler Maruyama
            V, T = euler_maruyama(sigma_noise, y0, tn, dt, n_neurons, n_clusters, K, params, stimulus)

        ######################################################################################################
        # VISUALIZATIONS

        # COMPUTATIONS FOR VISUALIZATIONS
        Vsum = []
        for i in range(len(V[0])):
            Vsum.append(0)
            for neuron in range(n_neurons * n_clusters):
                Vsum[i] += V[neuron][i]

        # SINGLE NEURONS
        colors = px.colors.qualitative.Plotly
        cluster = []
        for i in range(n_clusters):
            for j in range(n_neurons):
                cluster.append(i)
        fig_single = px.line(title='Single coupled neurons')
        for neuron in range(n_neurons * n_clusters):
            fig_single.add_scatter(x=T, y=V[neuron], mode='lines',
                                   line=dict(color=colors[cluster[neuron] % len(colors)]), showlegend=False)
        fig_single.update_xaxes(title_text='t [ms]')
        fig_single.update_yaxes(title_text='V_i [mV]')

        # SUM OF THE NEURONS
        fig_sum = px.line(y=Vsum, x=T, title='Sum of coupled the neurons')
        fig_sum.update_xaxes(title_text='t [ms]')
        fig_sum.update_yaxes(title_text='V [mV]')

        # PERIODOGRAM
        V_periodogram = []
        for index in range(len(Vsum)):
            if periodogram_a <= T[index] <= periodogram_b:
                V_periodogram.append(Vsum[index])
        fs = 1000 * len(V_periodogram) / tn
        f, Pxx = periodogram(detrend(V_periodogram), fs)
        kernel = np.array([1, 1, 1]) / 3  # Define the smoothing kernel
        smoothed_Pxx = np.convolve(Pxx, kernel, mode='same')
        f_smoothed = f[:len(smoothed_Pxx)]
        limit = 2000
        threshold = 0.1
        for i in range(round(limit / 2), len(f_smoothed)):
            if smoothed_Pxx[i] > threshold:
                limit = i * 5 + 100
        fig_periodogram = px.line(y=smoothed_Pxx, x=f_smoothed, title='Periodogram')
        fig_periodogram.update_layout(xaxis_range=[0, limit])
        fig_periodogram.update_xaxes(title_text='frequency [Hz]')
        fig_periodogram.update_yaxes(title_text='|P(f)|')

        # NETWORK VISUALIZATION
        fig_K = go.Figure(data=go.Heatmap(
            z=K,
            colorscale="purples",
            colorbar={"title": 'epsilon'}))
        fig_K.layout.height = 500
        fig_K.layout.width = 500
        fig_K.update_yaxes(autorange="reversed")
        fig_K.update_layout(
            title='Visualisation of the coupling matrix')

        # RETURN
        return [
            dcc.Graph(figure=fig_single),
            dcc.Graph(figure=fig_sum),
            html.Div(
                [html.Div(dcc.Graph(figure=fig_periodogram), style={'width': '49%', 'display': 'inline-block'}),
                 html.Div(dcc.Graph(figure=fig_K), style={'width': '49%', 'display': 'inline-block'})]
            ),
        ]
