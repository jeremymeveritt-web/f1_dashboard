"""Standalone CLI script for executing the ETL pipeline and training the ML model."""

import sys
import logging
import os
from src.config import SEASON_RANGE, JOLPICA_BASE_URL, MODEL_PATH
from src.api_client import JolpicaClient
from src.data_processor import build_historical_dataset, engineer_features
from src.predictor import PodiumPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting complete ETL and Model Training Pipeline...")
    
    # 1. Ingest Data
    logger.info(f"Fetching historical data for seasons: {SEASON_RANGE[0]} to {SEASON_RANGE[-1]}")
    client = JolpicaClient(JOLPICA_BASE_URL)
    raw_df = build_historical_dataset(client, SEASON_RANGE)
    
    if raw_df.empty:
        logger.error("Failed to fetch historical data. Exiting.")
        sys.exit(1)
        
    logger.info(f"Raw historical data shape: {raw_df.shape}")

    # 2. Transform & Engineer Features
    logger.info("Engineering features (rolling averages, affinities, etc.)...")
    processed_df = engineer_features(raw_df)
    logger.info(f"Processed data shape: {processed_df.shape}")

    # 3. Train Model
    logger.info("Training XGBoost Predictor...")
    predictor = PodiumPredictor(MODEL_PATH)
    metrics = predictor.train(processed_df)
    
    # 4. Report
    logger.info("=== Final Evaluation Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
            
    logger.info(f"Pipeline successfully finished. Model saved to {MODEL_PATH}")
    sys.exit(0)

if __name__ == "__main__":
    main()