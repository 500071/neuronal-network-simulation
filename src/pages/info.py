from dash import html, register_page, dcc
import dash_bootstrap_components as dbc

register_page(
    __name__,
    name='Info',
    top_nav=True,
    path='/info'
)


def layout():
    layout = html.Div([
        html.Br(),
        html.H1("neuronal models definitions"),
        "This website is used for neronal network simulations. It is possible to simulate networks on three models - "
        "the Morris-Lecar model, the Interneuron model and the Destexhe-Pare model. Here are the definitions of the "
        "three types of models. \n To run the simulations, go to the 'Morris-Lecar', 'Interneuron' or 'Destexhe-Paré' tabs.",
        html.Br(), html.Br(),
        html.Div(
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        html.Div([
                            "The Morris-Lecar neuron model is one of the basic models for describing neuronal dynamics. "
                            "The model was formulated in 1981 in a paper by Catherine Morris and Harold Lecar as a model"
                            " of the oscillatory behaviour of tension in the muscle tissue of a giant barnacle. Today, "
                            "however, it is commonly used as a model of human neuron dynamics. We formulate the model "
                            "as a differential equation:",
                            html.Br(),
                            dcc.Markdown(r'$$\begin{align}'
                                         r' C_i \frac{\mathrm{d}}{\mathrm{d}t} V_i &= I_{\text{ext}} - '
                                         r'g_\text{L} (V_i - V_\text{L}) - '
                                         r'g_{\text{Ca}} m_{\infty}(V_i) (V_i-V_{\text{Ca}}) - '
                                         r'g_\text{K} w_i(V_i-V_\text{K}) - \sum_{j=1}^N \varepsilon_{ij} (V_i - V_j),\\'
                                         r'\frac{\mathrm{d}}{\mathrm{d}t} w_i &= \frac{w_\infty (V_i) - w_i}{\tau_w (V_i)}.'
                                         r'\end{align}$$', mathjax=True),
                            dcc.Markdown(r"$$V_i$$ is the membrane potential of the $i$-th neuron and $$w_i$$ represents "
                                         r"the activation variable of K$^+$ channel of the $$i$$-th neuron, where $$i \in"
                                         r"\{1, \dots, N\}$$. $\varepsilon_{ij}$ is the strenght of coupling between "
                                         r"neurons $i$ and $j$.", mathjax=True),
                            html.H6("Adjustable parameters"),
                            dcc.Markdown(
                                r"$$C_i$$ is the membrane capacitance. It is random for each neuron and it follows "
                                r"truncated normal distribution $$C_i \sim TN(\mu, \sigma^2, a, b)$$, all parameters of the "
                                r"distribution can be specified by user. Parameters $g_\text{L}$, $g_\text{Ca}$ and "
                                r"$g_\text{K}$ represent the maximum leak, Ca$^{2+}$, and K$^{+}$ electrical conductances "
                                r"through membrane channels, respectively, and $V_\text{L}$, $V_\text{Ca}$, and "
                                r"$V_\text{K}$ are reversal potentials of the specific ion channels."
                                r"$I_\text{ext}$ is the externally applied DC current. The current can be static or a "
                                r"white noise with variance of $\sigma_\text{noise}^2$ can be added. An exponential "
                                r"stimulus in the form of $I_{\text{stimulus}} =  A \cdot \mathrm{e}^{- r (t-t_0)}$ where "
                                r"$t \in [t_{0}, t_{0} + T]$. Parameters of the stimulus $A, r, t_0$ are random for each neuron "
                                r"from truncated normal distribution with adjustable parameters", mathjax=True),

                        ]), title="Morris-Lecar model"
                    ),
                    dbc.AccordionItem(html.Div([
                            "The interneuron model was formulated in 1997 by John White and his colleagues at the Boston University. "
                            "The model was used to explain synchronous oscillations and loss of synchrony in a network "
                            "of inhibitory neurons. We formulate the model as a system of two differential equations:",
                            html.Br(),
                            dcc.Markdown(r'$$\begin{equation*}C_i \frac{\mathrm{d}}{\mathrm{d}t} V_i ='
                                         r' I_{\text{ext}} - I_\text{L} - I_{\text{Na}} - I_\text{K} -'
                                         r' \sum_{j=1}^N \varepsilon_{ij} (V_i - V_j)\end{equation*}$$', mathjax=True),
                            dcc.Markdown(r'where $$\begin{equation*}'
                                         r'I_{\text{L}} = g_{\text{L}} \cdot (V_i - V_{\text{L}}), \quad '
                                         r'I_{\text{Na}} = g_{\text{Na}} m_i^3 h_i \cdot (V_i - V_{\text{L}}), \quad '
                                         r'I_{\text{K}} = g_{\text{K}} n_i^4 \cdot (V_i - V_{\text{K}}),'
                                         r'\end{equation*}$$', mathjax=True),
                            dcc.Markdown(r"$$V_i$$ is the membrane potential of the $i$-th neuron and $$m_i$$, $$h_i$$, "
                                         r"$$n_i$$ represent the activation variable of K$^+$ and Na$^+$ channels of the"
                                         r" $$i$$-th neuron, where $$i \in \{1, \dots, N\}$$. $h_i$ and $n_i$ are given "
                                         r"by differencial equations. $\varepsilon_{ij}$ is the strenght of coupling between "
                                         r"neurons $i$ and $j$.", mathjax=True),

                            html.H6("Adjustable parameters"),
                            dcc.Markdown(
                                r"$$C_i$$ is the membrane capacitance. It is random for each neuron and it follows "
                                r"truncated normal distribution $$C_i \sim TN(\mu, \sigma^2, a, b)$$, all parameters of the "
                                r"distribution can be specified by user. Parameters $g_\text{Na}$ and "
                                r"$g_\text{K}$ represent the maximum Na$^{+}$ and K$^{+}$ electrical conductances "
                                r"through membrane channels, $V_\text{Na}$ and $V_\text{K}$, and "
                                r"$V_\text{K}$ are reversal potentials of the specific ion channels. "
                                r"$I_\text{ext}$ is the externally applied DC current. The current can be static or a "
                                r"white noise with variance of $\sigma_\text{noise}^2$ can be added. An exponential "
                                r"stimulus in the form of $I_{\text{stimulus}} =  A \cdot \mathrm{e}^{- r (t-t_0)}$ where "
                                r"$t \in [t_{0}, t_{0} + T]$. Parameters of the stimulus $A, r, t_0$ are random for each neuron "
                                r"from truncated normal distribution with adjustable parameters", mathjax=True),
                        ]), title="Interneuron model"
                    ),
                    dbc.AccordionItem(
                        html.Div([
                            "The last model to be defined is the Destexhe-Paré model of the neuron. The model was "
                            "published in 1999 by Alain Destexhe and Denis Paré. We formulate the model as a "
                            "differential equation:",
                            html.Br(),
                            dcc.Markdown(r'$$\begin{equation*}C_i \frac{\mathrm{d}}{\mathrm{d}t} V_i = I_{\text{ext}} '
                                         r'- I_\text{L} - I_{\text{Na}} - I_{\text{Kdr}} - I_\text{M} - '
                                         r'\sum_{j=1}^N \varepsilon_{ij} (V_i - V_j)\end{equation*}$$', mathjax=True),
                            dcc.Markdown(r'where $$\begin{equation*}'
                                         r' I_{\text{L}} = g_{\text{L}} \cdot (V_i - V_{\text{L}}), \quad '
                                         r' I_{\text{Na}} = g_{\text{Na}} m_i^3 h_i \cdot (V_i - V_{\text{L}}), \quad '
                                         r'I_{\text{Kdr}} = g_{\text{Kdr}} n_i^4 \cdot (V_i - V_{\text{K}}), \quad '
                                         r'I_{M} = g_{\text{M}} m_{M,i} \cdot (V_i - V_{\text{K}})'
                                         r'\end{equation*}$$', mathjax=True),
                            dcc.Markdown(r"$$V_i$$ is the membrane potential of the $i$-th neuron and $$m_i,h_i,n_i,"
                                         r"m_{Mi}$$ represent the activation variables of K$^+$ and Na$^+$ channels of the"
                                         r" $$i$$-th neuron, where $$i \in \{1, \dots, N\}$$. The activation variables"
                                         r" are given by differencial equations. $\varepsilon_{ij}$ is the strenght of "
                                         r"coupling between neurons $i$ and $j$.", mathjax=True),

                            html.H6("Adjustable parameters"),
                            dcc.Markdown(
                                r"$$C_i$$ is the membrane capacitance. It is random for each neuron and it follows "
                                r"truncated normal distribution $$C_i \sim TN(\mu, \sigma^2, a, b)$$, all parameters of the "
                                r"distribution can be specified by user. Parameters $g_\text{Na}$ and "
                                r"$g_\text{Kdr}$ represent the maximum Na$^{+}$ and K$^{+}$ electrical conductances "
                                r"through the membrane channels, $V_\text{Na}$ and $V_\text{K}$ are reversal potentials of "
                                r"the specific ion channels. $V_\text{T}$ and $V_\text{S}$ are parameters of the "
                                r"activation functions. "
                                r"$I_\text{ext}$ is the externally applied DC current. The current can be static or a "
                                r"white noise with variance of $\sigma_\text{noise}^2$ can be added. An exponential "
                                r"stimulus in the form of $I_{\text{stimulus}} =  A \cdot \mathrm{e}^{- r (t-t_0)}$ where "
                                r"$t \in [t_{0}, t_{0} + T]$. Parameters of the stimulus $A, r, t_0$ are random for each neuron "
                                r"from truncated normal distribution with adjustable parameters", mathjax=True),
                        ]), title="Destexhe-Paré model"
                    ),
                ],
                start_collapsed=True,
            ),
        )
    ], style={'margin-left': '110px', 'padding-right': '110px'})

    return layout
