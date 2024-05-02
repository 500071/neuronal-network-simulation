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
                            "as a system of two differential equations:",
                            html.Br(),
                            dcc.Markdown(r'$$\begin{align}'
                                         r' C_i \frac{\mathrm{d}}{\mathrm{d}t} V_i &= I_{\text{ext}} - '
                                         r'g_\text{L} (V_i - V_\text{L}) - '
                                         r'g_{\text{Ca}} m_{\infty}(V_i) (V_i-V_{\text{Ca}}) - '
                                         r'g_\text{K} w_i(V_i-V_\text{K}) - \sum_{j=1}^N \varepsilon_{ij} (V_i - V_j),\\'
                                         r'\frac{\mathrm{d}}{\mathrm{d}t} w_i &= \frac{w_\infty (V_i) - w_i}{\tau_w (V_i)}.'
                                         r'\end{align}$$', mathjax=True),
                            dcc.Markdown(r"$$V_i$$ is the membrane potential of the $i$-th neuron and $$w_i$$ represents "
                                         r"the activation variable fo K$^+$ channel of the $$i$$-th neuron, where $$i \in"
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

                        ]),
                        "This is the content of the first section", title="Morris-Lecar model"
                    ),
                    dbc.AccordionItem(
                        "This is the content of the second section", title="Interneuron model"
                    ),
                    dbc.AccordionItem(
                        "This is the content of the third section", title="Destexhe-Paré model"
                    ),
                ],
                start_collapsed=True,
            ),
        )
    ], style={'margin-left': '110px', 'padding-right': '110px'})

    return layout