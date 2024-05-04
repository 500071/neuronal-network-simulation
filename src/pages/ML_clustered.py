from dash import html, register_page, dcc  #, callback # If you need callbacks, import it here.
from dash import Input, Output, callback, State
from math import pow, sqrt
import scipy.integrate as integrate
import numpy as np
import random
import networkx as nx
from scipy.stats import truncnorm
import plotly.express as px
import plotly.graph_objects as go
import re
from scipy import signal
import dash_bootstrap_components as dbc

register_page(
    __name__,
    name='Morris-Lecar clustered',
    top_nav=True,
    path='/ML_clustered'
)

######################################################################################################
# SITE PRO SPRAZENI


def no_coupling(n):
    K = []
    for row in range(n):
        line = []
        for col in range(n):
            line.append(0)
        K.append(line)
    return np.asarray(K)


def small_world(n, k, p, seed):
    # n..pocet neuronu, k..s kolika neurony je propojen, p..pst prepojeni
    G = nx.watts_strogatz_graph(n, k, p, seed)
    return nx.adjacency_matrix(G).todense()


def small_world_outer(n, k, p, seed):
    np.random.seed(15)
    # pravidelna struktura
    K = np.zeros((n, n))
    for i in range(k):
        K = K + np.diag([1] * (n - i), i) + np.diag([1] * i, i - n)
    # prepojeni
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
    return np.ones((n, n))


def random_network(n, p, seed):
    G = nx.erdos_renyi_graph(n, p, seed, directed=False)  # random network
    return nx.adjacency_matrix(G).todense()


def random_network_outer(n, p, seed):
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
    K = np.zeros((n, n))
    K[n-1][0] = 1
    return np.asarray(K)

######################################################################################################
# MODEL NEURONU

def morris_lecar(t, variables, params, stimulus):
    # variables = [Vi, wi]
    # params = [C, Iext, gK, gCa, gL, VK, VCa, VL, C[neuron]
    # stimulus = [st_len, st_t0, st_A, st_r]

    Vi = variables[0]
    wi = variables[1]

    C = params[0]
    Iext = params[1]
    A = stimulus[2]
    r = stimulus[3]
    if stimulus[1] <= t <= stimulus[1] + stimulus[0]:
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

    # function
    minf = (1 + np.tanh((Vi - beta1) / beta2)) / 2
    winf = (1 + np.tanh((Vi - beta3) / beta4)) / 2
    tauinf = (1 / (phi * np.cosh((Vi - beta3) / (2 * beta4)))) / 2

    dVi = (-gCa * minf * (Vi - VCa) - gK * wi * (Vi - VK) - gL * (Vi - VL) + Iext) / C
    dwi = (winf - wi) / tauinf

    return [dVi, dwi]


def coupledML(t, variables, K, n_neurons, n_clusters, params, stimulus):
    # prvne vsechny V, pak vsechny N

    Vi = variables[:n_neurons*n_clusters]
    wi = variables[n_neurons*n_clusters:]
    dV = []
    dw = []

    for cluster in range(n_clusters):
        for neuron in range(n_neurons):
            params_neuron = [params[0][cluster][neuron]] # odpovidajici C
            for index in range(1, 8): #C, Iext, gK, gCa, gL, VK, VCa, VL
                params_neuron.append(params[index][cluster])
            stimulus_neuron = [stimulus[0][cluster]] # odpovidajici delka stimulu
            for index in range(1, 4):
                stimulus_neuron.append(stimulus[index][cluster][neuron])
            dVar = morris_lecar(t, [variables[neuron + cluster * n_neurons],
                                   variables[neuron + n_neurons * (n_clusters + cluster)]],
                               params_neuron, stimulus_neuron)
            dV.append(dVar[0])
            dw.append(dVar[1])

    for row in range(n_neurons * n_clusters):
        for col in range(n_neurons * n_clusters):
            dV[row] += (Vi[col] - Vi[row]) * K[row][col] / params[0][row % n_clusters][row // n_clusters]

    dV.extend(dw)
    return dV

######################################################################################################
# EULER-MARUYAMA METODA

def euler_maruyama(sigma_noise, X0, T, dt, n_neurons, n_clusters, K, params, stimulus):
    # dX =  f(X) dt + sigma dW
    # f...drift function (fce interneuron s parametry)
    # sigma_noise = sigma_noise / C
    # X0... pocatecni podminky
    # T... delka casoveho intervalu
    # dt... krok
    # N_eq... pocet neuronu
    #params = [C, Iext, gNa, gK, VNa, VK, C[0]]
    N_eq = n_neurons*n_clusters

    N = np.floor(T/dt).astype(int) # pocet kroku
    d = len(X0) # pocet rovnic v soustave celkem
    sigma = []
    for cluster in range(n_clusters):
        for neuron in range(n_neurons):
            sigma.append(sigma_noise / sqrt(params[0][cluster][neuron]))
    sigma_noise = sigma + [0] * (d - N_eq)  # sigma_noise + (d-N_eq) nul

    X = np.zeros((d, N + 1)) # pocet radku: pocet rovnic interneuronu, pocet sloupcu: pocet kroku+1
    X[:, 0] = X0 # prvni sloupec jsou pocatecni podminky
    t = np.arange(0, T+dt, dt)  #cas
    dW = np.vstack([np.sqrt(dt) * np.random.randn(N_eq, N), np.zeros((d - N_eq, N))]) # prvni radek prirustky Wienerova procesu, pak nuly

    for step in range(N): #pres vsechny kroky
        neuron = coupledML(t[step], X[:, step], K, n_neurons, n_clusters, params, stimulus)
        for i in range(len(neuron)):
            neuron[i] = neuron[i] * dt
        X[:, step + 1] += X[:, step] + neuron + sigma_noise * dW[:, step]
    return X, t


######################################################################################################
# NACITANI DAT

def input_parser(input_string, n, integer=False):
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
    # 1: no coupling, 2: all to all, 3: smallworld, 4: random network, 5: ring
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
    if sigma == 0:
        if n==1:
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
        html.H1("Morris-Lecar simulation — multiple clusters"),
        "Enter values for individual clusters separated by semicolons. Enter matrices in the form of [a;b];[c;d].",
        html.Br(),
        ######################################################################
        html.H5("Number of neurons and type of coupling"),
        html.Div([
            "n_clusters = ", dcc.Input(id='n_clusters', value=2.0, type='number', size='8'), " ",
            "n_neurons = ", dcc.Input(id='n_neurons', value=25.0, type='number', size='8'), " ",
        ], title="number of clusters and number of neurons in each cluster"),
        html.Br(),
        html.H6("coupling inside the clusters"),
        "1: no coupling, 2: all to all, 3: smallworld, 4: random network, 5: ring",
        html.Div([
            html.Div([
                " epsilon = ", dcc.Input(id='eps_clusters', value="0.001; 0.001", type='text', size='15'), " ",
            ], title="coupling strenght"),
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
            ], title="coupling strenght"),
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
                "V0 ∈ ( ", dcc.Input(id='V0beg', value="40; -40", type='text', size='15'),
                ", ", dcc.Input(id='V0end', value="40; -40", type='text', size='15'), " )",
            ], title="initial conditions for V0, random for each neuron from uniform distribution"),
            html.Span(' ', style={'display': 'inline-block', 'width': '30px'}),
            html.Div([
                " w0 = ", dcc.Input(id='w0', value="0.5; 0.5", type='text', size='15'), " ",
            ], title="initial conditions for the other variable, fixed for all neurons in the cluster"),
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
                "I_ext = ", dcc.Input(id='Iext', value="100; 100", type='text', size='15'),
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
            ], title="membrane capacitance, random for each neuron"),
        ], style=dict(display='flex')),
        html.Br(),
        html.Div([
            "g_K = ", dcc.Input(id='gK', value="8;8", type='text', size='15'),
            " g_Ca = ", dcc.Input(id='gCa', value="4;4", type='text', size='15'),
            " g_L = ", dcc.Input(id='gL', value="2;2", type='text', size='15'),
            " V_K = ", dcc.Input(id='VK', value="-80;-80", type='text', size='15'),
            " V_Ca = ", dcc.Input(id='VCa', value="120;120", type='text', size='15'),
            " V_L = ", dcc.Input(id='VL', value="-60;-60", type='text', size='15'),
        ], title="Morris-Lecar parameters"),
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
            ], title="start of the stimulus, random for each neuron"),
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
            ], title="amplitude of the stimulus, random for each neuron"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "r ∼ TN(",
                "μ = ", dcc.Input(id='st_r', value="0; 0", type='text', size='10'),
                " σ = ", dcc.Input(id='st_r_sigma', value="0; 0", type='text', size='10'),
                " [ ", dcc.Input(id='st_r_a', value="0; 0", type='text', size='10'),
                ", ", dcc.Input(id='st_r_b', value="0; 0", type='text', size='10'), " ])",
            ], title=" damping rate of the stimulus, random for each neuron"),
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
        html.Button(id='button_MLC', n_clicks=0, children="Redraw graph"),
        html.Div([], id='plots_MLC'),

    ], style={'margin-left': '110px'})
    return layout

######################################################################################################
# CALLBACK

@callback(
    Output(component_id='plots_MLC', component_property='children'),
    Input('button_MLC', 'n_clicks'),
    [State('n_clusters', 'value'), State('n_neurons', 'value'), State('eps_clusters', 'value'),
     State('coupling_clusters', 'value'), State('p_clusters', 'value'), State('k_clusters', 'value'),
     State('seed_clusters', 'value'),
     State('eps_outer', 'value'), State('coupling_outer', 'value'), State('p_outer', 'value'),
     State('k_outer', 'value'), State('seed_outer', 'value'), State('V0beg', 'value'), State('V0end', 'value'),
     State('w0', 'value'), State('t0', 'value'), State('tn', 'value'), State('dt', 'value'),
     State('Iext', 'value'), State('C_mu', 'value'), State('C_sigma', 'value'), State('C_a', 'value'),
     State('C_b', 'value'),
     State('gK', 'value'), State('gCa', 'value'), State('gL', 'value'), State('VK', 'value'), State('VCa', 'value'), State('VL', 'value'),
     State('st_t0', 'value'), State('st_t0_sigma', 'value'), State('st_t0_a', 'value'),
     State('st_t0_b', 'value'), State('st_tn', 'value'), State('st_A', 'value'), State('st_A_sigma', 'value'),
     State('st_A_a', 'value'), State('st_A_b', 'value'), State('st_r', 'value'), State('st_r_sigma', 'value'),
     State('st_r_a', 'value'), State('st_r_b', 'value'), State('sigma_noise', 'value'),
     State('periodogram_a', 'value'), State('periodogram_b', 'value')
     ]
)
def update_output(n_clicks, n_clusters, n_neurons, epsilon_clusters, coupling_clusters, p_clusters, k_clusters,
                      seed_clusters, epsilon_outer, coupling_outer, p_outer, k_outer, seed_outer, V0_beg, V0_end,
                      w0, t0, tn, dt, Iext, C_mu, C_sigma, C_a, C_b, gK, gCa, gL, VK, VCa, VL, st_t0_mu,
                      st_t0_sig, st_t0_a, st_t0_b, st_len, st_A_mu, st_A_sig, st_A_a, st_A_b, st_r_mu, st_r_sig,
                      st_r_a, st_r_b, sigma_noise, periodogram_a, periodogram_b
                      ):
    if n_clicks > 0:
        # DIAGONALA
        epsilon_clusters = input_parser(epsilon_clusters, n_clusters)
        coupling_clusters = input_parser(coupling_clusters, n_clusters, integer=True)
        k_clusters = input_parser(k_clusters, n_clusters, integer=True)
        p_clusters = input_parser(p_clusters, n_clusters)
        seed_clusters = input_parser(seed_clusters, n_clusters, integer=True)

        if any(str(param) == "input error" for param in
               [epsilon_clusters, coupling_clusters, k_clusters, p_clusters, seed_clusters]):
            return dbc.Alert("Input error!", color="danger")

        # TROJUHELNIKY
        epsilon = matrix_input_parser(epsilon_outer, n_clusters)
        coupling = matrix_input_parser(coupling_outer, n_clusters, integer=True)
        k = matrix_input_parser(k_outer, n_clusters, integer=True)
        p = matrix_input_parser(p_outer, n_clusters)
        seed = matrix_input_parser(seed_outer, n_clusters, integer=True)

        if any(str(param) == "input error" for param in [epsilon, coupling, k, p, seed]):
            return dbc.Alert("Input error!", color="danger")

        # MATICE VSECH PROMENNYCH
        np.fill_diagonal(epsilon, epsilon_clusters)
        np.fill_diagonal(coupling, coupling_clusters)
        np.fill_diagonal(k, k_clusters)
        np.fill_diagonal(p, p_clusters)
        np.fill_diagonal(seed, seed_clusters)

        # VYPOCET MATICE K
        K_list = []
        for row in range(n_clusters):
            line = []
            for col in range(n_clusters):
                if row == col:
                    K = epsilon[row][col] * \
                        generate_K(coupling[row][col], n_neurons, k[row][col], p[row][col], int(seed[row][col]))
                elif row < col:
                    K = epsilon[row][col] * \
                        generate_outer_K(coupling[row][col], n_neurons, k[row][col], p[row][col], int(seed[row][col]),
                                         "U")
                else:
                    K = epsilon[row][col] * \
                        generate_outer_K(coupling[row][col], n_neurons, k[row][col], p[row][col], int(seed[row][col]),
                                         "L")
                line.append(K)
            K_list.append(line)
        K = np.concatenate([np.concatenate(row, axis=1) for row in K_list], axis=0)

        # POCATECNI PODMINKY
        V0_beg = input_parser(V0_beg, n_clusters)
        V0_end = input_parser(V0_end, n_clusters)
        w0_input = input_parser(w0, n_clusters)
        V0 = []
        w0 = []
        for cluster in range(n_clusters):
            for neuron in range(n_neurons):
                V0.append(random.uniform(V0_beg[cluster], V0_end[cluster]))
                w0.append(w0_input[cluster])
        y0 = np.asarray(V0 + w0)

        # VOLBA PARAMETRU
        Iext = input_parser(Iext, n_clusters)
        C_mu = input_parser(C_mu, n_clusters)
        C_sigma = input_parser(C_sigma, n_clusters)
        C_a = input_parser(C_a, n_clusters)
        C_b = input_parser(C_b, n_clusters)
        C = []
        for cluster in range(n_clusters):
            C.append(
                random_truncnorm(C_a[cluster], C_b[cluster], C_mu[cluster], C_sigma[cluster], n_neurons))

        gK = input_parser(gK, n_clusters)
        gCa = input_parser(gCa, n_clusters)
        gL = input_parser(gL, n_clusters)
        VK = input_parser(VK, n_clusters)
        VCa = input_parser(VCa, n_clusters)
        VL = input_parser(VL, n_clusters)
        params = np.array([C, Iext, gK, gCa, gL, VK, VCa, VL])

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
        # RESENI
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

        # VYPOCTY PRO VIZUALIZACI
        Vsum = []
        for i in range(len(V[0])):
            Vsum.append(0)
            for neuron in range(n_neurons * n_clusters):
                Vsum[i] += V[neuron][i]

        ymin = min(Vsum)
        ymax = max(Vsum)
        for neuron in range(n_neurons * n_clusters):
            ymin = min(ymin, min(V[neuron]))
            ymax = max(ymax, max(V[neuron]))

        # NEURONY SAMOSTATNE
        colors = px.colors.qualitative.Plotly
        cluster = []
        for i in range(n_clusters):
            for j in range(n_neurons):
                cluster.append(i)
        fig_single = px.line(title='Single coupled neurons')
        for neuron in range(n_neurons * n_clusters):
            fig_single.add_scatter(x=T, y=V[neuron], mode='lines',
                                   line=dict(color=colors[cluster[neuron] % len(colors)]), showlegend=False)
            # fig_single.add_trace(px.line(x=T, y=V[neuron], line_color=colors[neuron % len(colors)]).data[0],)
        fig_single.update_xaxes(title_text='t [ms]')
        fig_single.update_yaxes(title_text='V_i [mV]')

        # SOUCET NEURONU
        fig_sum = px.line(y=Vsum, x=T, title='Sum of coupled the neurons')
        fig_sum.update_xaxes(title_text='t [ms]')
        fig_sum.update_yaxes(title_text='V [mV]')

        # PERIODOGRAM
        V_periodogram = []
        for index in range(len(Vsum)):
            if periodogram_a <= T[index] <= periodogram_b:
                V_periodogram.append(Vsum[index])
        fs = 1000 * len(V_periodogram) / tn
        f, Pxx = signal.periodogram(signal.detrend(V_periodogram), fs)
        kernel = np.array([1, 1, 1]) / 3  # Define the smoothing kernel
        smoothed_Pxx = np.convolve(Pxx, kernel, mode='same')
        f_smoothed = f[:len(smoothed_Pxx)]
        limit = 2000
        treshold = 0.1
        for i in range(round(limit / 2), len(f_smoothed)):
            if smoothed_Pxx[i] > treshold:
                limit = i * 5 + 100
        fig_periodogram = px.line(y=smoothed_Pxx, x=f_smoothed, title='Periodogram')
        fig_periodogram.update_layout(xaxis_range=[0, limit])
        fig_periodogram.update_xaxes(title_text='frequency [Hz]')
        fig_periodogram.update_yaxes(title_text='|P(f)|')

        # VIZUALIZACE SITE
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
