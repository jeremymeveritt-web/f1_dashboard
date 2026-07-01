"""Machine Learning pipeline utilizing XGBoost for podium probability scoring."""

import logging
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from src.config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

class PodiumPredictor:
    """Predicts podium finishes using historical F1 data."""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.scaler = None

    def train(self, df: pd.DataFrame) -> dict:
        """Trains the XGBClassifier and returns evaluation metrics."""
        logger.info("Initializing model training pipeline...")
        
        train_mask = df['season'].between(2018, 2023)
        test_mask = df['season'].between(2024, 2025)
        
        X_train = df[train_mask][FEATURE_COLUMNS]
        y_train = df[train_mask]['target']
        X_test = df[test_mask][FEATURE_COLUMNS]
        y_test = df[test_mask]['target']
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5,
            eval_metric='logloss',
            early_stopping_rounds=20
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_importances": self.get_feature_importances()
        }
        
        logger.info(f"Training Complete. F1: {metrics['f1']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        
        # Save artifacts
        joblib.dump({'model': self.model, 'scaler': self.scaler}, self.model_path)
        return metrics

    def load(self) -> None:
        """Loads trained model and scaler from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError("Run training first via train.py")
        
        artifacts = joblib.load(self.model_path)
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']

    def predict_proba(self, feature_row: pd.DataFrame) -> float:
        """Returns the probability [0.0, 1.0] of a podium finish."""
        try:
            if self.model is None or self.scaler is None:
                self.load()
            
            scaled_features = self.scaler.transform(feature_row[FEATURE_COLUMNS])
            prob = self.model.predict_proba(scaled_features)[0, 1]
            return float(prob)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0

    def get_feature_importances(self) -> dict[str, float]:
        """Returns sorted feature importance dictionary."""
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        feat_imp = dict(zip(FEATURE_COLUMNS, importances))
        return dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))