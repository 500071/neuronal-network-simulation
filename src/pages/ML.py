from dash import html, register_page, dcc  #, callback # If you need callbacks, import it here.
from dash import Input, Output, callback, State
import scipy.integrate as integrate
import numpy as np
import random
import networkx as nx
from scipy.stats import truncnorm
import plotly.express as px
import plotly.graph_objects as go
from scipy import signal

register_page(
    __name__,
    name='Morris-Lecar',
    top_nav=True,
    path='/ML'
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


def random_network(n, p, seed):
    G = nx.erdos_renyi_graph(n, p, seed, directed=False)  # random network
    return nx.adjacency_matrix(G).todense()


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

######################################################################################################
# MODEL NEURONU

def morris_lecar(t, variables, params, stimulus):
    # variables = [Vi, wi]
    # params = [C, Iext, gK, gCa, gL, VK, VCa, VL, C[neuron]
    # stimulus = [st_len, st_t0, st_A, st_r]

    Vi = variables[0]
    wi = variables[1]

    C = params[8]
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


def coupledML(t, variables, K, n, params, stimulus):
    # prvne vsechny V, pak vsechny N

    Vi = variables[:n]
    wi = variables[n:]
    dV = []
    dw = []

    for neuron in range(n):
        stimulus_neuron = [stimulus[0]]
        for index in range(1, 4):
            stimulus_neuron.append(stimulus[index][neuron])
            params[8] = params[0][neuron]
        dVar = morris_lecar(t, [variables[neuron], variables[neuron+n]], params, stimulus_neuron)
        dV.append(dVar[0])
        dw.append(dVar[1])


    for row in range(n):
        for col in range(n):
            dV[row] += (Vi[col] - Vi[row]) * K[row][col] / params[0][row]

    dV.extend(dw)
    return dV

######################################################################################################
# EULER-MARUYAMA METODA

def euler_maruyama(sigma_noise, X0, T, dt, N_eq, K, params, stimulus):
    # dX =  f(X) dt + sigma dW
    # f...drift function (fce interneuron s parametry)
    # sigma_noise = sigma_noice / C
    # X0... pocatecni podminky
    # T... delka casoveho intervalu
    # dt... krok
    # N_eq... pocet neuronu

    N = np.floor(T/dt).astype(int) # pocet kroku
    d = len(X0) # pocet rovnic v soustave celkem
    sigma_noise = [sigma_noise / x for x in params[0]] + [0]*(d-N_eq) # sigma_noise + (d-N_eq) nul
    X = np.zeros((d, N + 1)) # pocet radku: pocet rovnic interneuronu, pocet sloupcu: pocet kroku+1
    X[:, 0] = X0 # prvni sloupec jsou pocatecni podminky
    t = np.arange(0, T+dt, dt)  #cas
    dW = np.vstack([np.sqrt(dt) * np.random.randn(N_eq, N), np.zeros((d - N_eq, N))]) # prvni radek prirustky Wienerova procesu, pak nuly

    for step in range(N): #pres vsechny kroky
        neuron = coupledML(t[step], X[:, step], K, N_eq, params, stimulus)
        for i in range(len(neuron)):
            neuron[i] = neuron[i] * dt
        X[:, step + 1] += X[:, step] + neuron + sigma_noise * dW[:, step]
    return X, t

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
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n)



######################################################################################################
# LAYOUT

def layout():
    layout = html.Div([
        html.Br(),
        html.H1("Morris-Lecar model simulation"),
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
        html.Br(),

        ######################################################################
        html.H5("Initial conditions"),
        html.Div([
            html.Div([
                "V0 ∈ ( ", dcc.Input(id='V0beg', value=35, type='number', size='5'),
                ", ", dcc.Input(id='V0end', value=40, type='number', size='5'), " )",
            ], title="initial conditions for V0, random for each neuron from uniform distribution"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                " w0 = ", dcc.Input(id='w0', value=0.5, type='number', size='5'),
            ], title="initial conditions for the other variable, fixed for all neurons"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
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
                "I_ext = ", dcc.Input(id='Iext', value=100, type='number', size='5'),
            ], title="external DC current"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "C ∼ TN(",
                "μ = ", dcc.Input(id='C_mu', value=1, type='number', size='5'),
                " σ = ", dcc.Input(id='C_sigma', value=0.001, type='number', size='5'),
                " [ ", dcc.Input(id='C_a', value=0.95, type='number', size='5'),
                ", ", dcc.Input(id='C_b', value=1.05, type='number', size='5'), " ])",
            ], title="membrane capacitance, random for each neuron"),
        ], style=dict(display='flex')),
        html.Br(),
        html.Div([
            "g_K = ", dcc.Input(id='gK', value=8, type='number', size='5'),
            " g_Ca = ", dcc.Input(id='gCa', value=4, type='number', size='5'),
            " g_L = ", dcc.Input(id='gL', value=2, type='number', size='5'),
            " V_K = ", dcc.Input(id='VK', value=-80, type='number', size='5'),
            " V_Ca = ", dcc.Input(id='VCa', value=120, type='number', size='5'),
            " V_L = ", dcc.Input(id='VL', value=-60, type='number', size='5'),
        ], title="Morris-Lecar parameters"),
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
            ], title="start of the stimulus, random for each neuron"),
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
            ], title="amplitude of the stimulus, random for each neuron"),
            html.Span(' ', style={'display': 'inline-block', 'width': '10px'}),
            html.Div([
                "r ∼ TN(",
                "μ = ", dcc.Input(id='st_r', value=0, type='number', size='5'),
                " σ = ", dcc.Input(id='st_r_sigma', value=0, type='number', size='5'),
                " [ ", dcc.Input(id='st_r_a', value=0, type='number', size='5'),
                ", ", dcc.Input(id='st_r_b', value=0, type='number', size='5'), " ])",
            ], title=" damping rate of the stimulus, random for each neuron"),
        ], style=dict(display='flex')),
        html.Br(),

        ######################################################################
        html.H5("noise"),
        html.Div([
            "sigma_noise = ", dcc.Input(id='sigma_noise', value=0, type='number', size='5'),
        ], title="standard deviation of the noise"),
        html.Br(),

        ######################################################################
        html.Button(id='button_ML', n_clicks=0, children="Redraw graph"),
        html.Div([], id='plots_ML'),
    ], style={'margin-left': '110px'})
    return layout

######################################################################################################
# CALLBACK

@callback(
    Output(component_id='plots_ML', component_property='children'),
    Input('button_ML', 'n_clicks'),
    [State('n', 'value'), State('eps', 'value'), State('coupling-type', 'value'), State('p', 'value'),
     State('k', 'value'), State('seed', 'value'), State('V0beg', 'value'), State('V0end', 'value'),
     State('w0', 'value'), State('t0', 'value'), State('tn', 'value'), State('dt', 'value'), State('Iext', 'value'),
     State('st_t0', 'value'), State('st_t0_sigma', 'value'), State('st_t0_a', 'value'), State('st_t0_b', 'value'),
     State('st_tn', 'value'), State('st_A', 'value'), State('st_A_sigma', 'value'), State('st_A_a', 'value'),
     State('st_A_b', 'value'), State('st_r', 'value'), State('st_r_sigma', 'value'), State('st_r_a', 'value'),
     State('st_r_b', 'value'), State('C_mu', 'value'), State('C_sigma', 'value'), State('C_a', 'value'),
     State('C_b', 'value'), State('gK', 'value'), State('gCa', 'value'), State('gL', 'value'), State('VK', 'value'),
     State('VCa', 'value'), State('VL', 'value'), State('sigma_noise', 'value'),
     ]
)
def update_output_ML(n_clicks, n, epsilon, coupling_type, p, k, seed, V0_beg, V0_end, w0_init, t0, tn, dt, Iext,
                     st_t0_mu, st_t0_sig, st_t0_a, st_t0_b, st_len, st_A_mu, st_A_sig, st_A_a, st_A_b, st_r_mu,
                     st_r_sig, st_r_a, st_r_b, C_mu, C_sigma, C_a, C_b, gK, gCa, gL, VK, VCa, VL, sigma_noise):
    if n_clicks > 0:
        # GENEROVANI MATICE K
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

        # POCATECNI PODMINKY
        V0 = []
        w0 = []
        for i in range(n):
            V0.append(random.uniform(V0_beg, V0_end))
            w0.append(w0_init)
        y0 = V0 + w0

        # STIMULUS
        st_t0 = random_truncnorm(st_t0_a, st_t0_b, st_t0_mu, st_t0_sig, n)
        st_A = random_truncnorm(st_A_a, st_A_b, st_A_mu, st_A_sig, n)
        st_r = random_truncnorm(st_r_a, st_r_b, st_r_mu, st_r_sig, n)
        stimulus = [st_len, st_t0, st_A, st_r]

        # VOLBA PARAMETRU
        C = random_truncnorm(C_a, C_b, C_mu, C_sigma, n)
        params = [C, Iext, gK, gCa, gL, VK, VCa, VL, C[0]]

        # RESENI
        if sigma_noise == 0:
            # using Runge Kutta 45
            res = integrate.solve_ivp(coupledML, [t0, tn], y0, method='RK45', args=[K, n, params, stimulus])
            V = res.y
            T = res.t
        else:
            # using Euler Maruyama
            V, T = euler_maruyama(sigma_noise, y0, tn, dt, n, K, params, stimulus)

        # VYPOCTY PRO VIZUALIZACI
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

        # NEURONY SAMOSTATNE
        fig_single = px.line(title='Single coupled neurons')
        for neuron in range(n):
            # plt.plot(T, V[neuron], 'blue')
            fig_single.add_trace(px.line(x=T, y=V[neuron]).data[0])
        fig_single.update_xaxes(title_text='t [ms]')
        fig_single.update_yaxes(title_text='V_i [mV]')

        # SOUCET NEURONU
        fig_sum = px.line(y=Vsum, x=T, title='Sum of coupled the neurons')
        fig_sum.update_xaxes(title_text='t [ms]')
        fig_sum.update_yaxes(title_text='V [mV]')

        # PERIODOGRAM
        fs = 1000 * len(Vsum) / tn
        f, Pxx = signal.periodogram(signal.detrend(Vsum), fs)
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