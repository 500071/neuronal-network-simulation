# NEURONAL NETWORK SIMULATION APP - navbar definition
# AUTHOR: Markéta Trembaczová, 2024

import dash_bootstrap_components as dbc


def create_navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(
                dbc.NavLink(
                    "Info", href='/info',
                    target="_blank"
                )
            ),
            dbc.DropdownMenu(
                nav=True,
                in_navbar=True,
                label="Morris-Lecar",
                align_end=True,
                children=[
                    dbc.DropdownMenuItem("One cluster", href='/ML'),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Multiple clusters", href='/ML_clustered'),
                ],
            ),
            dbc.DropdownMenu(
                nav=True,
                in_navbar=True,
                label="Interneuron",
                align_end=True,
                children=[
                    dbc.DropdownMenuItem("One cluster", href='/IN'),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Multiple clusters", href='/IN_clustered'),
                ],
            ),
            dbc.DropdownMenu(
                nav=True,
                in_navbar=True,
                label="Destexhe-Paré",
                align_end=True,
                children=[
                    dbc.DropdownMenuItem("One cluster", href='/DP'),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Multiple clusters", href='/DP_clustered'),
                ],
            ),
        ],
        brand='Neuronal network simulation',
        brand_href="/",
        color="dark",
        dark=True,
    )

    return navbar
