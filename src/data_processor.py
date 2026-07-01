"""Data transformation, feature engineering, and signal processing."""

import pandas as pd
import numpy as np
from src.api_client import JolpicaClient
from src.cache_manager import get_cached_or_fetch

def build_historical_dataset(jolpica_client: JolpicaClient, seasons: list[int]) -> pd.DataFrame:
    """Iterates through seasons to build a concatenated master DataFrame."""
    def _fetch_all():
        dfs = []
        for season in seasons:
            df = jolpica_client.get_season_results(season)
            if not df.empty:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    return get_cached_or_fetch(f"historical_dataset_{min(seasons)}_{max(seasons)}", _fetch_all, ttl_hours=168)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers predictive features from raw race results.
    
    Args:
        df: Raw historical DataFrame.
    Returns:
        DataFrame enriched with all required FEATURE_COLUMNS and target.
    """
    df = df.copy()
    
    # a) driver_track_affinity: mean finish pos per driver/circuit
    df['driver_track_affinity'] = df.groupby(['driver_id', 'circuit_id'])['position'].transform('mean')
    
    # b) driver_form_3race: rolling 3-race avg points
    df = df.sort_values(['season', 'round'])
    df['driver_form_3race'] = (
        df.groupby('driver_id')['points']
        .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    ).fillna(0)
    
    # c) constructor_quali_gap: estimated gap to pole based on grid
    df['constructor_quali_gap'] = (df['grid'] - 1).clip(lower=0) * 0.15
    
    # d) grid_position_normalized
    df['grid_position_normalized'] = df['grid'] / 20.0
    
    # e) season_progress
    df['season_progress'] = df['round'] / df.groupby('season')['round'].transform('max')
    
    # f) is_wet_race: binary proxy based on accident/spun off rates (>30%)
    df['is_wet_race'] = (
        df.groupby(['season', 'round'])['status']
        .transform(lambda x: (x.isin(['Accident', 'Spun off']).sum() / len(x)) > 0.3)
    ).astype(int)
    
    # g) target: 1 if podium, 0 otherwise
    df['target'] = (df['position'] <= 3).astype(int)
    
    # Fill any remaining NaNs safely
    df = df.fillna(0)
    return df
   
def prepare_prediction_input(driver_id: str, circuit_id: str, grid_pos: int, historical_df: pd.DataFrame) -> pd.DataFrame:
    """Constructs a single-row feature vector for live race prediction."""
    # Base defaults
    feats = {
        'driver_track_affinity': 10.0,
        'driver_form_3race': 0.0,
        'constructor_quali_gap': (grid_pos - 1) * 0.15,
        'grid_position_normalized': grid_pos / 20.0,
        'season_progress': 0.5,
        'is_wet_race': 0
    }
    
    # Try to extract real historical affinity and form
    driver_history = historical_df[historical_df['driver_id'] == driver_id]
    if not driver_history.empty:
        feats['driver_form_3race'] = driver_history.sort_values(['season', 'round'])['points'].tail(3).mean()
        
        track_history = driver_history[driver_history['circuit_id'] == circuit_id]
        if not track_history.empty:
            feats['driver_track_affinity'] = track_history['position'].mean()
            
    return pd.DataFrame([feats])

def process_telemetry_for_chart(telemetry_df: pd.DataFrame, lap_number: int) -> pd.DataFrame:
    """Resamples telemetry to uniform 10m intervals for performant plotting."""
    if telemetry_df.empty or 'Distance' not in telemetry_df.columns:
        return pd.DataFrame()
        
    df = telemetry_df[telemetry_df['LapNumber'] == lap_number].copy() if 'LapNumber' in telemetry_df.columns else telemetry_df.copy()
    if df.empty:
         return df

    # Drop duplicates for interpolation
    df = df.drop_duplicates(subset=['Distance'])
    
    max_dist = df['Distance'].max()
    if pd.isna(max_dist):
        return pd.DataFrame()
        
    uniform_distances = np.arange(0, max_dist, 10)
    
    # Interpolate numeric columns onto uniform distance grid
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    resampled_df = pd.DataFrame({'Distance': uniform_distances})
    
    for col in numeric_cols:
        if col != 'Distance':
            resampled_df[col] = np.interp(uniform_distances, df['Distance'], df[col])
            
    return resampled_df