# PAGE FOR SINGLE CLUSTER DESTEXHE-PARÉ SIMULATION
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

register_page(
    __name__,
    name='Destexhe-Paré',
    top_nav=True,
    path='/DP'
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
            if col == (row - 1) % n or col == (row + 1) % n:
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
# LAYOUT

def layout():
    layout = html.Div([
        html.Br(),
        html.H1("Destexhe-Paré model simulation"),
        ######################################################################
        html.H5("Number of neurons and type of coupling"),
        html.Div([
            html.Div([
                "n = ", dcc.Input(id='n', value=5.0, type='number', size='8'), " ",
            ], title="number of neurons"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " epsilon = ", dcc.Input(id='eps', value=0.02, type='number', size='8'),
            ], title="coupling strenght"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                dcc.Dropdown(
                    id='coupling-type',
                    options=['no coupling', 'all to all', 'small world', 'random', 'ring'],
                    value='no coupling',
                    placeholder='Choose the coupling type...',
                    style={'width': '80%'}
                )
            ], style={'width': '35%', 'display': 'inline-block'}, title="coupling type"),
            html.Div([
                " p = ", dcc.Input(id='p', type='number', size='6', value=0.5),
            ], title="random networks: probability of edge creation \n"
                     "small world networks: probability of rewiring the edges"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " k = ", dcc.Input(id='k', type='number', size='6', value=3)
            ], title="small world networks: initial number of neighbours of each node before the rewiring the edges"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " seed = ", dcc.Input(id='seed', type='number', size='6', value=123)
            ], title="seed for random and small world networks"),
        ], style=dict(display='flex')),

        ######################################################################
        html.Br(),
        html.H5("Initial conditions"),
        html.Div([
            html.Div([
                "V0 ∈ ( ", dcc.Input(id='V0beg', value=0, type='number', size='5'),
                ", ", dcc.Input(id='V0end', value=5, type='number', size='5'), " )",
            ], title="initial conditions for V0, random for each neuron from uniform distribution"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " m0 = ", dcc.Input(id='m0', value=0.3, type='number', size='5'),
                " h0 = ", dcc.Input(id='h0', value=0.42, type='number', size='5'),
                " n0 = ", dcc.Input(id='n0', value=0.3, type='number', size='5'),
                " nM0 = ", dcc.Input(id='nM0', value=0.4, type='number', size='5'),
            ], title="initial conditions for the other variables, fixed for all neurons"),
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
        html.Div([html.Div([
                "I_ext = ", dcc.Input(id='Iext', value=43, type='number', size='5'),
            ], title="external DC current"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "sigma_noise = ", dcc.Input(id='sigma_noise', value=0, type='number', size='5'),
            ], title="standard deviation of the noise"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "C ∼ TN(",
                "μ = ", dcc.Input(id='C_mu', value=1, type='number', size='5'),
                " σ = ", dcc.Input(id='C_sigma', value=0.001, type='number', size='5'),
                " [ ", dcc.Input(id='C_a', value=0.95, type='number', size='5'),
                ", ", dcc.Input(id='C_b', value=1.05, type='number', size='5'), " ])",
            ], title="membrane capacitance, random from truncated normal distribution with given parameters for each "
                     "neuron"),
        ], style=dict(display='flex')),
        html.Br(),
        html.Div([
            "g_Na = ", dcc.Input(id='gNa', value=120, type='number', size='5'),
            " g_K = ", dcc.Input(id='gK', value=100, type='number', size='5'),
            " V_Na = ", dcc.Input(id='VNa', value=55, type='number', size='5'),
            " V_K = ", dcc.Input(id='VK', value=-85, type='number', size='5'),
            " V_T = ", dcc.Input(id='VT', value=-58, type='number', size='5'),
            " V_S = ", dcc.Input(id='VS', value=-10, type='number', size='5')
        ], title="Destexhe-Paré parameters"),
        html.Br(),

        ######################################################################
        html.H5("STIMULUS"),
        html.Div([
            html.Div([
                "t0_st ∼ TN(",
                "μ = ", dcc.Input(id='st_t0', value=50, type='number', size='5'),
                " σ = ", dcc.Input(id='st_t0_sigma', value=0, type='number', size='5'),
                " [ ", dcc.Input(id='st_t0_a', value=0, type='number', size='5'),
                ", ", dcc.Input(id='st_t0_b', value=0, type='number', size='5'), " ])",
            ], title="start of the stimulus, random from truncated normal distribution with given parameters for each "
                     "neuron"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "T_st = ", dcc.Input(id='st_tn', value=20, type='number', size='5')
            ], title="length of the stimulus"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
        ], style=dict(display='flex')),
        html.Br(),
        html.Div([
            html.Div([
                "A ∼ TN(",
                "μ = ", dcc.Input(id='st_A', value=0, type='number', size='5'),
                " σ = ", dcc.Input(id='st_A_sigma', value=0, type='number', size='5'),
                " [ ", dcc.Input(id='st_A_a', value=0, type='number', size='5'),
                ", ", dcc.Input(id='st_A_b', value=0, type='number', size='5'), " ])",
            ], title="amplitude of the stimulus, random from truncated normal distribution with given parameters for "
                     "each neuron"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "r ∼ TN(",
                "μ = ", dcc.Input(id='st_r', value=0, type='number', size='5'),
                " σ = ", dcc.Input(id='st_r_sigma', value=0, type='number', size='5'),
                " [ ", dcc.Input(id='st_r_a', value=0, type='number', size='5'),
                ", ", dcc.Input(id='st_r_b', value=0, type='number', size='5'), " ])",
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
        html.Button(id='button_DP', n_clicks=0, children="Redraw graph"),
        html.Div([], id='plots_DP'),
    ], style={'margin-left': '110px'})
    return layout

######################################################################################################
# CALLBACK
# function for what happens after clicking on the button


@callback(
    Output(component_id='plots_DP', component_property='children'),
    Input('button_DP', 'n_clicks'),
    [State('n', 'value'), State('eps', 'value'), State('coupling-type', 'value'), State('p', 'value'),
     State('k', 'value'), State('seed', 'value'), State('V0beg', 'value'), State('V0end', 'value'),
     State('m0', 'value'), State('h0', 'value'), State('n0', 'value'), State('nM0', 'value'), State('t0', 'value'),
     State('tn', 'value'), State('dt', 'value'), State('Iext', 'value'), State('st_t0', 'value'),
     State('st_t0_sigma', 'value'), State('st_t0_a', 'value'), State('st_t0_b', 'value'), State('st_tn', 'value'),
     State('st_A', 'value'), State('st_A_sigma', 'value'), State('st_A_a', 'value'), State('st_A_b', 'value'),
     State('st_r', 'value'), State('st_r_sigma', 'value'), State('st_r_a', 'value'), State('st_r_b', 'value'),
     State('C_mu', 'value'), State('C_sigma', 'value'), State('C_a', 'value'), State('C_b', 'value'),
     State('gNa', 'value'), State('gK', 'value'), State('VNa', 'value'), State('VK', 'value'), State('VT', 'value'),
     State('VS', 'value'), State('sigma_noise', 'value'), State('periodogram_a', 'value'),
     State('periodogram_b', 'value')
     ]
)
def update_output_DP(n_clicks, n, epsilon, coupling_type, p, k, seed, V0_beg, V0_end, m0_init, h0_init, n0_init,
                     nM0_init, t0, tn, dt, Iext, st_t0_mu, st_t0_sig, st_t0_a, st_t0_b, st_len, st_A_mu, st_A_sig,
                     st_A_a, st_A_b, st_r_mu, st_r_sig, st_r_a, st_r_b, C_mu, C_sigma, C_a, C_b, gNa, gK, VNa, VK, VT,
                     VS, sigma_noise, periodogram_a, periodogram_b):
    if n_clicks > 0:
        # GENERATE COUPLING MATRIX K
        if coupling_type == 'all to all':
            K = all_to_all(n)
        elif coupling_type == 'small world':
            K = small_world(n, k, p, seed)
        elif coupling_type == 'random':
            K = random_network(n, p, seed)
        elif coupling_type == 'ring':
            K = ring(n)
        else:
            K = no_coupling(n)
        K = K * epsilon

        # INITIAL CONDITIONS
        V0 = []
        m0 = []
        h0 = []
        n0 = []
        nM0 = []
        for i in range(n):
            V0.append(random.uniform(V0_beg, V0_end))
            m0.append(m0_init)
            h0.append(h0_init)
            n0.append(n0_init)
            nM0.append(nM0_init)
        y0 = V0 + m0 + h0 + n0 + nM0

        # STIMULUS
        st_t0 = random_truncnorm(st_t0_a, st_t0_b, st_t0_mu, st_t0_sig, n)
        st_A = random_truncnorm(st_A_a, st_A_b, st_A_mu, st_A_sig, n)
        st_r = random_truncnorm(st_r_a, st_r_b, st_r_mu, st_r_sig, n)
        stimulus = [st_len, st_t0, st_A, st_r]
        print(st_r)

        # PARAMETERS
        C = random_truncnorm(C_a, C_b, C_mu, C_sigma, n)
        params = [C, Iext, gNa, gK, VNa, VK, VT, VS, C[0]]

        # SOLUTION
        if sigma_noise == 0:  # standard deviation of the added noise
            # using Runge Kutta 45
            res = integrate.solve_ivp(coupled_DP, [t0, tn], y0, method='RK45', args=[K, n, params, stimulus])
            V = res.y
            T = res.t
        else:
            # using Euler Maruyama
            V, T = euler_maruyama(sigma_noise, y0, tn, dt, n, K, params, stimulus)

        # COMPUTATIONS FOR VISUALIZATIONS
        Vsum = []
        for i in range(len(V[0])):
            Vsum.append(0)
            for neuron in range(n):
                Vsum[i] += V[neuron][i]

        # SINGLE NEURONS
        fig_single = px.line(title='Single coupled neurons')
        for neuron in range(n):
            fig_single.add_trace(px.line(x=T, y=V[neuron]).data[0])
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
        kernel = np.array([1, 1, 1]) / 3
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
