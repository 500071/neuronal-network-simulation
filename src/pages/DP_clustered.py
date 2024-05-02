from dash import html, register_page  #, callback # If you need callbacks, import it here.

register_page(
    __name__,
    name='Destexhe-Paré clustered',
    top_nav=True,
    path='/DP_clustered'
)


def layout():
    layout = html.Div([
        html.Br(),
        html.H1("simulace více shluků Destexhe-Paré modelu"),
    ], style={'margin-left': '110px'})
    return layout