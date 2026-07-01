"""External API clients for Jolpica (historical data) and FastF1 (telemetry)."""

import time
import logging
import requests
import pandas as pd
import fastf1 as ff1
from typing import Optional
from pathlib import Path
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

class JolpicaClient:
    """Client for the Jolpica F1 historical data API."""
    
    def __init__(self, base_url: str, rate_limit_delay: float = 0.5):
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay

    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Helper to make rate-limited requests to Jolpica."""
        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(f"{self.base_url}/{endpoint}.json", params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"Jolpica API request failed: {e}")
            return {}

    def get_race_results(self, season: int, round_num: int) -> dict:
        """Fetches raw race results for a specific season and round."""
        return self._make_request(f"{season}/{round_num}/results")

    def get_season_results(self, season: int) -> pd.DataFrame:
        """Paginates through an entire season to build a unified DataFrame of results."""
        all_results = []
        limit = 30
        offset = 0
        total = float('inf')

        while offset < total:
            data = self._make_request(f"{season}/results", params={"limit": limit, "offset": offset})
            if not data:
                break
                
            mrdata = data.get("MRData", {})
            total = int(mrdata.get("total", 0))
            races = mrdata.get("RaceTable", {}).get("Races", [])
            
            for race in races:
                round_num = int(race.get("round"))
                circuit_id = race.get("Circuit", {}).get("circuitId")
                for result in race.get("Results", []):
                    fastest_lap = result.get("FastestLap", {})
                    all_results.append({
                        "season": season,
                        "round": round_num,
                        "circuit_id": circuit_id,
                        "driver_id": result.get("Driver", {}).get("driverId"),
                        "constructor_id": result.get("Constructor", {}).get("constructorId"),
                        "grid": int(result.get("grid", 0)),
                        "position": int(result.get("positionText", 99) if str(result.get("positionText", "")).isdigit() else 99),
                        "points": float(result.get("points", 0.0)),
                        "status": result.get("status"),
                        "fastest_lap_rank": int(fastest_lap.get("rank", 99)),
                        "fastest_lap_time_ms": fastest_lap.get("Time", {}).get("time") 
                    })
            offset += limit
            
        return pd.DataFrame(all_results)

    def get_driver_standings(self, season: int, round_num: int) -> pd.DataFrame:
        """Fetches driver standings after a specific round."""
        data = self._make_request(f"{season}/{round_num}/driverStandings")
        return pd.DataFrame(data)

    def get_constructor_standings(self, season: int, round_num: int) -> pd.DataFrame:
        """Fetches constructor standings after a specific round."""
        data = self._make_request(f"{season}/{round_num}/constructorStandings")
        return pd.DataFrame(data)


class FastF1Client:
    """Client for fetching high-resolution telemetry via FastF1."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def enable_cache(self) -> None:
        """Enables local filesystem caching for FastF1 requests."""
        ff1.Cache.enable_cache(str(self.cache_dir))

    def get_session(self, year: int, gp: str, session_type: str) -> Optional[ff1.core.Session]:
        """Loads a specific session's core data."""
        try:
            session = ff1.get_session(year, gp, session_type)
            session.load()
            return session
        except Exception as e:
            logger.warning(f"Failed to load FastF1 session ({year} {gp} {session_type}): {e}")
            return None

    def get_lap_telemetry(self, session: ff1.core.Session, driver_code: str) -> pd.DataFrame:
        """Extracts granular telemetry (Speed, Throttle, etc.) for a driver's laps."""
        try:
            laps = session.laps.pick_driver(driver_code)
            if laps.empty:
                return pd.DataFrame()
            return laps.get_telemetry()
        except Exception as e:
            logger.warning(f"Failed to get telemetry for {driver_code}: {e}")
            return pd.DataFrame()

    def get_all_driver_laps(self, session: ff1.core.Session) -> pd.DataFrame:
        """Returns summarized lap data for all drivers in a session."""
        try:
            return session.laps
        except Exception as e:
            logger.warning(f"Failed to get all driver laps: {e}")
            return pd.DataFrame()