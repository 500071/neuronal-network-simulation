from dash import html, register_page  #, callback # If you need callbacks, import it here.

register_page(
    __name__,
    name='Home',
    top_nav=True,
    path='/'
)


def layout():
    layout = html.Div([
        html.Br(),
        html.H1("Neuronal network simulation"),
        "This website was created for the purposes of research activities of the ",
        "Nonlinear Dynamics Team at the Department of Mathematics and Statistics, Faculty of Science, Masaryk ",
        "University. This website was created as a part of the GAMU Interdisciplinary MUNI/G/1213/2022 project on ",
        "Mathematical modeling of very and ultra-fast oscillations in EEG signals",
        html.Br(),
        "This website is a tool for modelling of complex networks of neurons. It supports three models of neurons — ",
        "Morris-Lecar model, interneuron model and Destexhe-Paré model. We model coupling of neurons using random ",
        "networks, small world networks, ring networks and all to all networks. We allow coupling of the neurons into "
        "one or more coupled clusters.",
        html.Br(),
        "To see the definitions of the models, see the 'INFO' tab. To run the simulations, go to the 'MORRIS-LECAR', "
        "'INTERNEURON' and 'DESTEXHE-PARÉ' tabs."
    ], style={'margin-left': '110px', 'padding-right': '110px'})
    return layout
