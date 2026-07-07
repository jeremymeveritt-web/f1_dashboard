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

DEFAULT_CONSTRUCTOR_COLOR = "#8A8A93"

logger = logging.getLogger(__name__)

# Initialize clients
ff1_client = FastF1Client(CACHE_DIR)
ff1_client.enable_cache()
jolpica_client = JolpicaClient(JOLPICA_BASE_URL)
predictor = PodiumPredictor(MODEL_PATH)

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#F4F6FB', family='Inter, sans-serif', size=12),
    margin=dict(l=40, r=20, t=40, b=30),
    xaxis=dict(showgrid=True, gridcolor='rgba(0,212,255,0.08)', zeroline=False, linecolor='rgba(0,212,255,0.2)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(0,212,255,0.08)', zeroline=False, linecolor='rgba(0,212,255,0.2)'),
    hoverlabel=dict(bgcolor='#12121A', bordercolor='#00D4FF', font=dict(color='#F4F6FB', family='JetBrains Mono, monospace')),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8A8A93')),
    transition=dict(duration=400, easing='cubic-in-out')
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

    # Helper: map a FastF1 driver code (e.g. "VER") to its team's real hex color.
    def get_driver_color(session, driver_code, fallback):
        try:
            if session is not None and driver_code:
                team_color = ff1.plotting.get_driver_color(driver_code, session=session)
                if team_color:
                    return team_color
        except Exception as e:
            logger.debug(f"Could not resolve team color for {driver_code}: {e}")
        return fallback

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

        fallback_colors = {'A': '#00D4FF', 'B': '#FF1801'}

        for d_idx, d_code in [('A', driver_a), ('B', driver_b)]:
            if not d_code: continue
            raw_tel = ff1_client.get_lap_telemetry(session, d_code)
            tel = process_telemetry_for_chart(raw_tel, lap)

            if tel.empty: continue

            driver_color = get_driver_color(session, d_code, fallback_colors[d_idx])

            fig_speed.add_trace(go.Scatter(x=tel['Distance'], y=tel['Speed'], mode='lines', name=f"{d_code}", line=dict(color=driver_color)))
            fig_tb.add_trace(go.Scatter(x=tel['Distance'], y=tel['Throttle'], fill='tozeroy', name=f"{d_code} Thr", line=dict(color=driver_color)))
            fig_tb.add_trace(go.Scatter(x=tel['Distance'], y=tel['Brake']*100, mode='lines', name=f"{d_code} Brk", line=dict(color='#8A8A93', dash='dot')))
            fig_gear.add_trace(go.Scatter(x=tel['Distance'], y=tel['Gear'], mode='lines', line_shape='hv', name=f"{d_code} Gear", line=dict(color=driver_color)))

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
            if historical_df.empty:
                return fig_prob, fig_feat, html.Div(
                    "No historical data available. Check your network connection.",
                    style={'color': '#FF1801'}
                )

            latest_season = historical_df['season'].max()

            # Filter to current active drivers
            latest_season_df = historical_df[historical_df['season'] == latest_season]
            if latest_season_df.empty:
                latest_season_df = historical_df[historical_df['season'] == (latest_season - 1)]

            active_drivers = latest_season_df['driver_id'].dropna().unique()

            # Map GP accurately to Jolpica's circuit ID format
            target_race = latest_season_df[latest_season_df['round'] == int(round_num)]
            if not target_race.empty:
                circuit_id = target_race['circuit_id'].iloc[0]
            else:
                circuit_id = gp.split(" ")[0].lower()  # Fallback for un-run races

            # Calculate a realistic synthetic grid position based on average finish this season
            driver_stats = latest_season_df.groupby('driver_id')['position'].mean().sort_values()

            # Most recent constructor per driver, for team-accurate bar coloring
            driver_constructor = (
                latest_season_df.sort_values(['round'])
                .groupby('driver_id')['constructor_id']
                .last()
                .to_dict()
            )

            probs = []
            driver_names = []
            bar_colors = []

            for driver_id in active_drivers:
                # Estimate starting position: rank based on real average finish this season
                try:
                    grid_pos = driver_stats.index.get_loc(driver_id) + 1
                except KeyError:
                    grid_pos = 10  # Default midfield if data is missing

                feat_row = prepare_prediction_input(driver_id, circuit_id, grid_pos, historical_df)
                prob = predictor.predict_proba(feat_row)

                probs.append(prob * 100)
                driver_names.append(str(driver_id).replace("_", " ").title())
                constructor_id = driver_constructor.get(driver_id)
                bar_colors.append(CONSTRUCTOR_COLORS.get(constructor_id, DEFAULT_CONSTRUCTOR_COLOR))

            # Sort for horizontal bar chart (ascending so highest probability lands on top)
            sorted_data = sorted(zip(probs, driver_names, bar_colors), key=lambda row: row[0])
            y_drivers = [row[1] for row in sorted_data]
            x_probs = [row[0] for row in sorted_data]
            sorted_colors = [row[2] for row in sorted_data]

            fig_prob.add_trace(go.Bar(
                x=x_probs, y=y_drivers, orientation='h', marker_color=sorted_colors
            ))
            fig_prob.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title="Probability (%)")

            feats = predictor.get_feature_importances()
            fig_feat.add_trace(go.Bar(
                x=list(feats.values()), y=list(feats.keys()), orientation='h', marker_color='#00D4FF'
            ))
            fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'})

            metrics_div = html.Div([
                html.Div([html.Span("Status: "), html.Span("Ready", style={'color': '#00D4FF'})], className="metric-row"),
                html.Div([html.Span("Drivers scored: "), html.Span(str(len(active_drivers)))], className="metric-row"),
                html.Div([html.Span("Circuit: "), html.Span(str(circuit_id))], className="metric-row"),
            ])

            return fig_prob, fig_feat, metrics_div

        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}")
            return fig_prob, fig_feat, html.Div(
                f"Prediction failed: {e}", style={'color': '#FF1801'}
            )

    # 4. Lap Time Delta (Driver A vs Driver B lap-by-lap comparison)
    @app.callback(
        Output('lap-delta-chart', 'figure'),
        Input('load-session-btn', 'n_clicks'),
        [State('driver-a-dropdown', 'value'),
         State('driver-b-dropdown', 'value'),
         State('season-dropdown', 'value'),
         State('gp-dropdown', 'value'),
         State('session-type-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_lap_delta(n_clicks, driver_a, driver_b, season, gp, session_type):
        fig = go.Figure(layout=DARK_LAYOUT)
        fig.update_layout(title="Lap Time Delta (A - B)", xaxis_title="Lap", yaxis_title="Delta (s)")

        if not all([season, gp, session_type, driver_a, driver_b]):
            return fig

        session = ff1_client.get_session(int(season), gp, session_type)
        if session is None:
            return fig

        try:
            laps_a = session.laps.pick_driver(driver_a)[['LapNumber', 'LapTime']].dropna()
            laps_b = session.laps.pick_driver(driver_b)[['LapNumber', 'LapTime']].dropna()

            merged = pd.merge(laps_a, laps_b, on='LapNumber', suffixes=('_a', '_b'))
            if merged.empty:
                return fig

            merged['delta_s'] = (merged['LapTime_a'] - merged['LapTime_b']).dt.total_seconds()

            color_a = get_driver_color(session, driver_a, '#00D4FF')
            bar_colors = [color_a if v <= 0 else '#FF1801' for v in merged['delta_s']]

            fig.add_trace(go.Bar(
                x=merged['LapNumber'], y=merged['delta_s'],
                marker_color=bar_colors,
                name=f"{driver_a} vs {driver_b}"
            ))
            fig.add_hline(y=0, line_color='#555', line_width=1)
            fig.update_layout(
                title=f"Lap Time Delta: {driver_a} (negative = faster) vs {driver_b}"
            )
            return fig
        except Exception as e:
            logger.warning(f"Failed to compute lap delta: {e}")
            return fig

    # 5. Circuit Speed Map (broadcast-style track outline colored by speed)
    @app.callback(
        Output('circuit-map-chart', 'figure'),
        Input('lap-slider', 'value'),
        [State('driver-a-dropdown', 'value'),
         State('driver-b-dropdown', 'value'),
         State('season-dropdown', 'value'),
         State('gp-dropdown', 'value'),
         State('session-type-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_circuit_map(lap, driver_a, driver_b, season, gp, session_type):
        fig = go.Figure(layout=DARK_LAYOUT)
        fig.update_layout(
            title="Circuit Speed Map",
            xaxis=dict(visible=False, scaleanchor='y', showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            plot_bgcolor='#0A0A0F'
        )

        driver_code = driver_a or driver_b
        if not all([season, gp, session_type]) or not driver_code:
            return fig

        session = ff1_client.get_session(int(season), gp, session_type)
        if session is None:
            return fig

        try:
            raw_tel = ff1_client.get_lap_telemetry(session, driver_code)
            if raw_tel.empty or 'X' not in raw_tel.columns or 'Y' not in raw_tel.columns:
                return fig

            tel = raw_tel[raw_tel['LapNumber'] == lap].copy() if 'LapNumber' in raw_tel.columns else raw_tel.copy()
            if tel.empty:
                tel = raw_tel.copy()

            fig.add_trace(go.Scatter(
                x=tel['X'], y=tel['Y'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=tel['Speed'],
                    colorscale='Turbo',
                    colorbar=dict(title="km/h"),
                    showscale=True
                ),
                name=driver_code
            ))
            fig.update_layout(title=f"Circuit Speed Map - {driver_code} (Lap {lap})")
            return fig
        except Exception as e:
            logger.warning(f"Failed to build circuit map for {driver_code}: {e}")
            return fig
