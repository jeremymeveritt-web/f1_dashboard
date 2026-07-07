"""Dash App Entry Point and Layout definitions."""
import os
import sys
import dash
from dash import html, dcc
from src.config import SEASON_RANGE, SESSION_TYPES

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap"
    ],
    suppress_callback_exceptions=True
)
app.title = "F1 Predictive Analytics"


def build_telemetry_tab() -> html.Div:
    return html.Div(className="grid-container fade-in", children=[
        # Row 1
        html.Div(className="f1-card col-4", children=[
            html.Div(className="card-header", children=[
                html.Span(className="card-icon"), html.H3("SESSION SELECTOR")
            ]),
            html.Label("SEASON", className="field-label"),
            dcc.Dropdown(id='season-dropdown', options=[{'label': y, 'value': y} for y in SEASON_RANGE], placeholder="Season", className="f1-dropdown"),
            html.Label("GRAND PRIX", className="field-label"),
            dcc.Dropdown(id='gp-dropdown', placeholder="Grand Prix", className="f1-dropdown"),
            html.Label("SESSION TYPE", className="field-label"),
            dcc.Dropdown(id='session-type-dropdown', options=[{'label': s, 'value': s} for s in SESSION_TYPES], placeholder="Session Type", className="f1-dropdown"),
            html.Hr(className="divider"),
            html.Label("DRIVER A", className="field-label driver-a-label"),
            dcc.Dropdown(id='driver-a-dropdown', placeholder="Driver A", className="f1-dropdown"),
            html.Label("DRIVER B", className="field-label driver-b-label"),
            dcc.Dropdown(id='driver-b-dropdown', placeholder="Driver B", className="f1-dropdown"),
            html.Button("LOAD SESSION", id='load-session-btn', className="btn-primary")
        ]),
        html.Div(className="f1-card col-8", children=[
            html.Div(className="card-header", children=[
                html.Span(className="card-icon"), html.H3("LAP TIME DELTA")
            ]),
            dcc.Loading(dcc.Graph(id='lap-delta-chart', config={'displayModeBar': False}), type="circle", color="#00D4FF")
        ]),
        # Row 2
        html.Div(className="f1-card col-12", children=[
            html.Div(className="card-header", children=[
                html.Span(className="card-icon"), html.H3("SPEED TRACE")
            ]),
            dcc.Loading(dcc.Graph(id='speed-trace-chart', config={'displayModeBar': False}), type="circle", color="#00D4FF"),
            html.Div(className="slider-wrap", children=[
                dcc.Slider(id='lap-slider', min=1, max=50, step=1, value=1, marks=None, tooltip={"placement": "bottom", "always_visible": True})
            ])
        ]),
        # Row 3
        html.Div(className="f1-card col-6", children=[
            html.Div(className="card-header", children=[html.Span(className="card-icon"), html.H3("THROTTLE / BRAKE")]),
            dcc.Loading(dcc.Graph(id='throttle-brake-chart', config={'displayModeBar': False}), type="circle", color="#00D4FF")
        ]),
        html.Div(className="f1-card col-6", children=[
            html.Div(className="card-header", children=[html.Span(className="card-icon"), html.H3("GEAR TRACE")]),
            dcc.Loading(dcc.Graph(id='gear-trace-chart', config={'displayModeBar': False}), type="circle", color="#00D4FF")
        ])
    ])


def build_predictor_tab() -> html.Div:
    return html.Div(className="grid-container fade-in", children=[
        # Row 1
        html.Div(className="f1-card col-4", children=[
            html.Div(className="card-header", children=[
                html.Span(className="card-icon"), html.H3("RACE CONFIGURATION")
            ]),
            html.Label("UPCOMING GRAND PRIX", className="field-label"),
            dcc.Dropdown(id='pred-gp-dropdown', placeholder="Upcoming Grand Prix", className="f1-dropdown"),
            html.Label("CHAMPIONSHIP ROUND", className="field-label"),
            dcc.Input(id='pred-round-input', type='number', placeholder="Round #", className="f1-input"),
            html.Button("RUN PREDICTION", id='run-pred-btn', className="btn-primary btn-accent")
        ]),
        html.Div(className="f1-card col-8", children=[
            html.Div(className="card-header", children=[
                html.Span(className="card-icon"), html.H3("PODIUM PROBABILITY")
            ]),
            dcc.Loading(dcc.Graph(id='podium-prob-chart', config={'displayModeBar': False}), type="circle", color="#00D4FF")
        ]),
        # Row 2
        html.Div(className="f1-card col-6", children=[
            html.Div(className="card-header", children=[html.Span(className="card-icon"), html.H3("MODEL FEATURE WEIGHTS")]),
            dcc.Loading(dcc.Graph(id='feature-importance-chart', config={'displayModeBar': False}), type="circle", color="#00D4FF")
        ]),
        html.Div(className="f1-card col-6", children=[
            html.Div(className="card-header", children=[html.Span(className="card-icon"), html.H3("MODEL PERFORMANCE")]),
            html.Div(id='metrics-card', children=[html.P("Model not trained.", className="text-muted")])
        ]),
        # Row 3
        html.Div(className="f1-card col-12", children=[
            html.Div(className="card-header", children=[
                html.Span(className="card-icon"), html.H3(id='circuit-map-title', children="CIRCUIT SPEED MAP")
            ]),
            dcc.Loading(dcc.Graph(id='circuit-map-chart', config={'displayModeBar': False}), type="circle", color="#00D4FF")
        ])
    ])


app.layout = html.Div(className="app-shell", children=[
    html.Div(className="bg-grid"),
    html.Header(className="app-header", children=[
        html.Div(className="header-left", children=[
            html.Div(className="logo-mark"),
            html.Div(children=[
                html.H1("F1 TELEMETRY & PREDICTOR", className="app-title"),
                html.Span("PREDICTIVE TELEMETRY & RACE ANALYSIS SYSTEM", className="app-subtitle")
            ])
        ]),
        html.Div(className="header-right", children=[
            html.Div(className="status-pill", children=[
                html.Span(className="status-dot"),
                html.Span("LIVE SYSTEM", className="status-text")
            ])
        ])
    ]),
    dcc.Tabs(className="f1-tabs", value="tab-telemetry", children=[
        dcc.Tab(label="TELEMETRY ANALYSIS", value="tab-telemetry", className="f1-tab", selected_className="f1-tab--selected", children=[build_telemetry_tab()]),
        dcc.Tab(label="RACE PREDICTOR", value="tab-predictor", className="f1-tab", selected_className="f1-tab--selected", children=[build_predictor_tab()])
    ])
])

if __name__ == '__main__':
    from callbacks import register_callbacks
    register_callbacks(app)
    app.run(debug=True)
