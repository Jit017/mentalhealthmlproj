#!/usr/bin/env python3
"""
Simple Setup Script for Streamlit App
Trains essential models quickly for the web application
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import joblib
import os

def main():
    print("🚀 Setting up models for Streamlit app...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv('digital_habits_vs_mental_health.csv')
    
    # Create target variables
    print("🎯 Creating target variables...")
    df['high_stress'] = (df['stress_level'] >= 7).astype(int)
    
    # Create features
    print("🔧 Creating features...")
    df['digital_wellness_score'] = (
        (24 - df['screen_time_hours']) / 24 * 10 +
        (5 - df['social_media_platforms_used']) / 5 * 10 +
        (10 - df['hours_on_Reels']) / 10 * 10
    ) / 3
    
    df['sleep_quality'] = np.where(
        (df['sleep_hours'] >= 7) & (df['sleep_hours'] <= 9),
        1, 0
    )
    
    df['screen_sleep_ratio'] = df['screen_time_hours'] / df['sleep_hours']
    df['social_media_intensity'] = df['social_media_platforms_used'] * df['hours_on_Reels']
    df['stress_mood_imbalance'] = abs(df['stress_level'] - (10 - df['mood_score']))
    
    # Select features
    features = [
        'screen_time_hours', 'social_media_platforms_used', 'hours_on_Reels',
        'sleep_hours', 'digital_wellness_score', 'sleep_quality',
        'screen_sleep_ratio', 'social_media_intensity', 'stress_mood_imbalance'
    ]
    
    # Train Random Forest for stress
    print("🌲 Training Random Forest...")
    X = df[features]
    y_stress = df['high_stress']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_stress, test_size=0.2, random_state=42, stratify=y_stress
    )
    
    rf_stress = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_stress.fit(X_train, y_train)
    
    # Train XGBoost for mood (using stress as proxy for now)
    print("🚀 Training XGBoost...")
    xgb_mood = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_mood.fit(X_train, y_train)  # Using same target for simplicity
    
    # Train K-Means clustering
    print("🎯 Training K-Means...")
    lifestyle_features = [
        'screen_time_hours', 'social_media_platforms_used', 'hours_on_Reels',
        'sleep_hours', 'digital_wellness_score', 'sleep_quality',
        'screen_sleep_ratio', 'social_media_intensity'
    ]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[lifestyle_features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Train Isolation Forest
    print("🔍 Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_scaled)
    
    # Save models
    print("💾 Saving models...")
    joblib.dump(rf_stress, 'models/rf_stress.joblib')
    joblib.dump(xgb_mood, 'models/xgb_mood.joblib')
    joblib.dump(kmeans, 'models/kmeans.joblib')
    joblib.dump(iso_forest, 'models/isolation_forest.joblib')
    
    # Save preprocessing components
    preprocessing_components = {
        'scaler': scaler,
        'feature_lists': {
            'lifestyle': lifestyle_features,
            'stress': features,
            'mood': features
        }
    }
    joblib.dump(preprocessing_components, 'models/preprocessing.joblib')
    
    # Save model performance
    performance = {
        'rf_stress': {
            'accuracy': rf_stress.score(X_test, y_test),
            'feature_importance': dict(zip(features, rf_stress.feature_importances_))
        },
        'xgb_mood': {
            'accuracy': xgb_mood.score(X_test, y_test),
            'feature_importance': dict(zip(features, xgb_mood.feature_importances_))
        }
    }
    joblib.dump(performance, 'models/model_performance.joblib')
    
    print("✅ Models saved successfully!")
    print("🎉 Ready to start Streamlit app!")
    print("Run: streamlit run app.py")

if __name__ == "__main__":
    main()
