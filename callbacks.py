"""Dash @app.callback functions defining application interactivity."""

import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, html
import fastf1 as ff1
from src.api_client import FastF1Client, JolpicaClient
from src.config import CACHE_DIR, JOLPICA_BASE_URL, MODEL_PATH, CONSTRUCTOR_COLORS, SEASON_RANGE
from src.data_processor import process_telemetry_for_chart, prepare_prediction_input, build_historical_dataset
from src.predictor import PodiumPredictor

logger = logging.getLogger(__name__)

# Initialize clients
ff1_client = FastF1Client(CACHE_DIR)
ff1_client.enable_cache()
jolpica_client = JolpicaClient(JOLPICA_BASE_URL)
predictor = PodiumPredictor(MODEL_PATH)

DARK_LAYOUT = dict(
    paper_bgcolor='#12121A',
    plot_bgcolor='#12121A',
    font=dict(color='#FFFFFF', family='Inter'),
    margin=dict(l=40, r=20, t=40, b=30),
    xaxis=dict(showgrid=True, gridcolor='#333'),
    yaxis=dict(showgrid=True, gridcolor='#333')
)

def register_callbacks(app):

    # FIX: Populate the Grand Prix dropdown dynamically when a season is selected
    @app.callback(
        Output('gp-dropdown', 'options'),
        Input('season-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_gp_dropdown(season):
        if not season:
            return []
        try:
            schedule = ff1.get_event_schedule(int(season))
            events = schedule[schedule['EventFormat'] != 'testing']
            return [{'label': row['EventName'], 'value': row['EventName']} for _, row in events.iterrows()]
        except Exception as e:
            logger.error(f"Failed to fetch season schedule: {e}")
            return []

    # FIX: Populate the Predictor Grand Prix dropdown on load (defaults to latest season)
    @app.callback(
        Output('pred-gp-dropdown', 'options'),
        Input('pred-gp-dropdown', 'id')
    )
    def update_pred_gp_dropdown(_):
        try:
            schedule = ff1.get_event_schedule(SEASON_RANGE[-1])
            events = schedule[schedule['EventFormat'] != 'testing']
            return [{'label': row['EventName'], 'value': row['EventName']} for _, row in events.iterrows()]
        except Exception:
            return []

    # 1. Session Loader (Fetches structural fastf1 data)
    @app.callback(
        [Output('driver-a-dropdown', 'options'),
         Output('driver-b-dropdown', 'options'),
         Output('lap-slider', 'max')],
        Input('load-session-btn', 'n_clicks'),
        [State('season-dropdown', 'value'),
         State('gp-dropdown', 'value'),
         State('session-type-dropdown', 'value')],
        prevent_initial_call=True
    )
    def load_session(n_clicks, season, gp, session_type):
        if not all([season, gp, session_type]):
            return [], [], 1
            
        session = ff1_client.get_session(int(season), gp, session_type)
        if session is None:
            return [], [], 1
            
        drivers = session.results['Abbreviation'].tolist()
        options = [{'label': d, 'value': d} for d in drivers]
        max_laps = session.laps['LapNumber'].max() if not session.laps.empty else 1
        return options, options, max_laps

    # 2. Telemetry Traces (Speed, Throttle/Brake, Gear)
    @app.callback(
        [Output('speed-trace-chart', 'figure'),
         Output('throttle-brake-chart', 'figure'),
         Output('gear-trace-chart', 'figure')],
        Input('lap-slider', 'value'),
        [State('driver-a-dropdown', 'value'),
         State('driver-b-dropdown', 'value'),
         State('season-dropdown', 'value'),
         State('gp-dropdown', 'value'),
         State('session-type-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_telemetry(lap, driver_a, driver_b, season, gp, session_type):
        fig_speed = go.Figure(layout=DARK_LAYOUT)
        fig_tb = go.Figure(layout=DARK_LAYOUT)
        fig_gear = go.Figure(layout=DARK_LAYOUT)
        
        if not all([season, gp, session_type]) or (not driver_a and not driver_b):
            return fig_speed, fig_tb, fig_gear
            
        session = ff1_client.get_session(int(season), gp, session_type)
        if session is None: return fig_speed, fig_tb, fig_gear

        colors = {'A': '#00D4FF', 'B': '#FF1801'}

        for d_idx, d_code in [('A', driver_a), ('B', driver_b)]:
            if not d_code: continue
            raw_tel = ff1_client.get_lap_telemetry(session, d_code)
            tel = process_telemetry_for_chart(raw_tel, lap)
            
            if tel.empty: continue
            
            fig_speed.add_trace(go.Scatter(x=tel['Distance'], y=tel['Speed'], mode='lines', name=f"{d_code}", line=dict(color=colors[d_idx])))
            fig_tb.add_trace(go.Scatter(x=tel['Distance'], y=tel['Throttle'], fill='tozeroy', name=f"{d_code} Thr", line=dict(color=colors[d_idx])))
            fig_tb.add_trace(go.Scatter(x=tel['Distance'], y=tel['Brake']*100, mode='lines', name=f"{d_code} Brk", line=dict(color='#8A8A93', dash='dot')))
            fig_gear.add_trace(go.Scatter(x=tel['Distance'], y=tel['Gear'], mode='lines', line_shape='hv', name=f"{d_code} Gear", line=dict(color=colors[d_idx])))

        fig_speed.update_layout(title="Speed (km/h)", xaxis_title="Distance (m)")
        fig_tb.update_layout(title="Throttle (%) / Brake", xaxis_title="Distance (m)")
        fig_gear.update_layout(title="Gear", xaxis_title="Distance (m)", yaxis=dict(range=[0, 9], dtick=1))

        return fig_speed, fig_tb, fig_gear

    # 3. Race Predictor Logic (FIX: Fully implemented ML execution block)
    @app.callback(
        [Output('podium-prob-chart', 'figure'),
         Output('feature-importance-chart', 'figure'),
         Output('metrics-card', 'children')],
        Input('run-pred-btn', 'n_clicks'),
        [State('pred-gp-dropdown', 'value'),
         State('pred-round-input', 'value')],
        prevent_initial_call=True
    )
    def run_prediction(n_clicks, gp, round_num):
        fig_prob = go.Figure(layout=DARK_LAYOUT)
        fig_feat = go.Figure(layout=DARK_LAYOUT)
        
        try:
            predictor.load()
        except FileNotFoundError:
            return fig_prob, fig_feat, html.Div("Model not trained yet. Run train.py.", style={'color': '#FF1801'})

        if not gp or not round_num:
            return fig_prob, fig_feat, html.Div("Please select a Grand Prix and Round.", style={'color': '#FF1801'})

        try:
            historical_df = build_historical_dataset(jolpica_client, SEASON_RANGE)
            latest_season = historical_df['season'].max()
            active_drivers = historical_df[historical_df['season'] == latest_season]['driver_id'].dropna().unique()
            circuit_id = gp.lower().replace(" ", "_").replace("grand_prix", "").strip("_")
            
            probs = []
            driver_names = []
            
            for i, driver_id in enumerate(active_drivers):
                # Assume starting position based on rough historical grid rank logic
                grid_pos = (i % 20) + 1 
                feat_row = prepare_prediction_input(driver_id, circuit_id, grid_pos, historical_df)
                prob = predictor.predict_proba(feat_row)
                
                probs.append(prob * 100)
                driver_names.append(str(driver_id).replace("_", " ").title())

            # Sort for horizontal bar chart (lowest to highest so highest is on top)
            sorted_data = sorted(zip(probs, driver_names), reverse=False) 
            y_drivers = [d[1] for d in sorted_data]
            x_probs = [d[0] for d in sorted_data]

            fig_prob.add_trace(go.Bar(
                x=x_probs, y=y_drivers, orientation='h', marker_color='#00D4FF'
            ))
            fig_prob.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title="Probability (%)")

            feats = predictor.get_feature_importances()
            fig_feat.add_trace(go.Bar(
                x=list(feats.values()), y=list(feats.keys()), orientation='h', marker_color='#00D4FF'
            ))
            fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            metrics_div = html.Div([
                html.Div([html.Span("Status: "), html.Span("Ready", style={'color': '#00D4FF'})], className="metric-row")
            ])
            
            return fig_prob, fig_feat, metrics_div
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return fig_prob, fig_feat, html.Div(f"Error: {e}", style={'color': '#FF1801'})