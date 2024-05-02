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
        "Nonlinear Dynamics Team at the Department of Mathematics and Statistics, Faculty of Science, Masaryk University."
    ], style={'margin-left': '110px', 'padding-right': '110px'})
    return layout
