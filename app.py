"""Dash App Entry Point and Layout definitions."""
import os
import sys
import dash
from dash import html, dcc
from src.config import SEASON_RANGE, SESSION_TYPES

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap"
    ],
    suppress_callback_exceptions=True
)
app.title = "F1 Predictive Analytics"

def build_telemetry_tab() -> html.Div:
    return html.Div(className="grid-container", children=[
        # Row 1
        html.Div(className="f1-card col-4", children=[
            html.H3("SESSION SELECTOR"),
            dcc.Dropdown(id='season-dropdown', options=[{'label': y, 'value': y} for y in SEASON_RANGE], placeholder="Season"),
            dcc.Dropdown(id='gp-dropdown', placeholder="Grand Prix"),
            dcc.Dropdown(id='session-type-dropdown', options=[{'label': s, 'value': s} for s in SESSION_TYPES], placeholder="Session Type"),
            html.Hr(),
            dcc.Dropdown(id='driver-a-dropdown', placeholder="Driver A (Cyan)"),
            dcc.Dropdown(id='driver-b-dropdown', placeholder="Driver B (Red)"),
            html.Button("LOAD SESSION", id='load-session-btn', className="btn-primary")
        ]),
        html.Div(className="f1-card col-8", children=[
            html.H3("LAP TIME DELTA"),
            dcc.Loading(dcc.Graph(id='lap-delta-chart'))
        ]),
        # Row 2
        html.Div(className="f1-card col-12", children=[
            html.H3("SPEED TRACE"),
            dcc.Graph(id='speed-trace-chart'),
            html.Div(style={'padding': '0 20px'}, children=[
                dcc.Slider(id='lap-slider', min=1, max=50, step=1, value=1, marks=None, tooltip={"placement": "bottom", "always_visible": True})
            ])
        ]),
        # Row 3
        html.Div(className="f1-card col-6", children=[dcc.Graph(id='throttle-brake-chart')]),
        html.Div(className="f1-card col-6", children=[dcc.Graph(id='gear-trace-chart')])
    ])

def build_predictor_tab() -> html.Div:
    return html.Div(className="grid-container", children=[
        # Row 1
        html.Div(className="f1-card col-4", children=[
            html.H3("RACE CONFIGURATION"),
            dcc.Dropdown(id='pred-gp-dropdown', placeholder="Upcoming Grand Prix"),
            dcc.Input(id='pred-round-input', type='number', placeholder="Championship Round", style={'width': '100%', 'padding': '8px', 'marginBottom': '10px'}),
            html.Button("RUN PREDICTION", id='run-pred-btn', className="btn-primary")
        ]),
        html.Div(className="f1-card col-8", children=[
            html.H3("PODIUM PROBABILITY"),
            dcc.Graph(id='podium-prob-chart')
        ]),
        # Row 2
        html.Div(className="f1-card col-6", children=[
            html.H3("MODEL FEATURE WEIGHTS"),
            dcc.Graph(id='feature-importance-chart')
        ]),
        html.Div(className="f1-card col-6", children=[
            html.H3("MODEL PERFORMANCE"),
            html.Div(id='metrics-card', children=[html.P("Model not trained.", className="text-muted")])
        ]),
        # Row 3
        html.Div(className="f1-card col-12", children=[
            html.H3(id='circuit-map-title', children="CIRCUIT SPEED MAP"),
            dcc.Loading(dcc.Graph(id='circuit-map-chart'))
        ])
    ])

app.layout = html.Div([
    html.H1("F1 TELEMETRY & PREDICTOR DASHBOARD", style={'textAlign': 'center', 'color': 'var(--accent-cyan)'}),
    dcc.Tabs([
        dcc.Tab(label="TELEMETRY ANALYSIS", children=[build_telemetry_tab()], style={'backgroundColor': '#12121A'}, selected_style={'backgroundColor': '#00D4FF', 'color': '#000'}),
        dcc.Tab(label="RACE PREDICTOR", children=[build_predictor_tab()], style={'backgroundColor': '#12121A'}, selected_style={'backgroundColor': '#00D4FF', 'color': '#000'})
    ])
])

if __name__ == '__main__':
    from callbacks import register_callbacks
    register_callbacks(app)
    app.run(debug=True)