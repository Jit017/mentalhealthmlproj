#!/usr/bin/env python3
"""
Digital Habits vs Mental Health - Enhanced Streamlit App
Main application file with improved UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Digital Habits vs Mental Health",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .cluster-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .anomaly-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .recommendation-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f093fb 0%, #f5576c 100%);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the original dataset"""
    return pd.read_csv('digital_habits_vs_mental_health.csv')

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        models = {}
        models['rf_stress'] = joblib.load('models/rf_stress.joblib')
        models['xgb_mood'] = joblib.load('models/xgb_mood.joblib')
        models['kmeans'] = joblib.load('models/kmeans.joblib')
        models['isolation_forest'] = joblib.load('models/isolation_forest.joblib')
        
        preprocessing = joblib.load('models/preprocessing.joblib')
        models['scaler'] = preprocessing['scaler']
        models['feature_lists'] = preprocessing['feature_lists']
        
        models['performance'] = joblib.load('models/model_performance.joblib')
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def create_features(user_input):
    """Create features from user input"""
    screen_time = user_input['screen_time_hours']
    social_media = user_input['social_media_platforms_used']
    reels_time = user_input['hours_on_Reels']
    sleep_hours = user_input['sleep_hours']
    
    digital_wellness_score = (
        (24 - screen_time) / 24 * 10 +
        (5 - social_media) / 5 * 10 +
        (10 - reels_time) / 10 * 10
    ) / 3
    
    sleep_quality = 1 if (sleep_hours >= 7 and sleep_hours <= 9) else 0
    screen_sleep_ratio = screen_time / sleep_hours if sleep_hours > 0 else 0
    social_media_intensity = social_media * reels_time
    stress_mood_imbalance = abs(user_input['stress_level'] - (10 - user_input['mood_score']))
    
    features = {
        'screen_time_hours': screen_time,
        'social_media_platforms_used': social_media,
        'hours_on_Reels': reels_time,
        'sleep_hours': sleep_hours,
        'digital_wellness_score': digital_wellness_score,
        'sleep_quality': sleep_quality,
        'screen_sleep_ratio': screen_sleep_ratio,
        'social_media_intensity': social_media_intensity,
        'stress_mood_imbalance': stress_mood_imbalance
    }
    
    return features

def predict_mental_health(user_input, models):
    """Make predictions"""
    features = create_features(user_input)
    
    # Create DataFrames with proper feature names
    stress_features_df = pd.DataFrame([features], columns=models['feature_lists']['stress'])
    lifestyle_features_df = pd.DataFrame([features], columns=models['feature_lists']['lifestyle'])
    
    stress_pred = models['rf_stress'].predict(stress_features_df)[0]
    stress_proba = models['rf_stress'].predict_proba(stress_features_df)[0]
    
    mood_pred = models['xgb_mood'].predict(stress_features_df)[0]
    mood_proba = models['xgb_mood'].predict_proba(stress_features_df)[0]
    
    lifestyle_scaled = models['scaler'].transform(lifestyle_features_df)
    cluster_pred = models['kmeans'].predict(lifestyle_scaled)[0]
    
    anomaly_score = models['isolation_forest'].decision_function(lifestyle_scaled)[0]
    is_anomaly = models['isolation_forest'].predict(lifestyle_scaled)[0] == -1
    
    return {
        'stress_prediction': stress_pred,
        'stress_probability': stress_proba,
        'mood_prediction': mood_pred,
        'mood_probability': mood_proba,
        'cluster': cluster_pred,
        'anomaly_score': anomaly_score,
        'is_anomaly': is_anomaly,
        'features': features
    }

def create_digital_wellness_gauge(score):
    """Create a gauge chart for digital wellness score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Digital Wellness Score"},
        delta = {'reference': 5},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgray"},
                {'range': [3, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(importance_dict):
    """Create feature importance chart"""
    df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
    df = df.sort_values('Importance', ascending=True)
    
    fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                 title="Feature Importance for Stress Prediction",
                 color='Importance', color_continuous_scale='viridis')
    fig.update_layout(height=400)
    return fig

def main():
    # Header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Digital Habits vs Mental Health</h1>
        <h3>Interactive Mental Health Analysis & Prediction</h3>
        <p>Discover how your digital habits impact your mental well-being</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and models
    data = load_data()
    models = load_models()
    
    if models is None:
        st.error("❌ Failed to load models. Please run the setup first.")
        st.info("Run: python simple_setup.py")
        return
    
    # Sidebar navigation with improved styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; color: white; text-align: center;">
        <h3>📊 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Predictions", "📈 Analysis", "📊 Data Explorer", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page(data, models)
    elif page == "🔮 Predictions":
        show_predictions_page(models)
    elif page == "📈 Analysis":
        show_analysis_page(data, models)
    elif page == "📊 Data Explorer":
        show_data_explorer_page(data)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(data, models):
    st.markdown("## 🏠 Welcome to Digital Habits vs Mental Health Analysis")
    
    # Key metrics with gradient cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Records</h3>
            <h2>{len(data):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Features</h3>
            <h2>{len(data.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'rf_stress' in models['performance']:
            acc = models['performance']['rf_stress']['accuracy']
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Stress Model</h3>
                <h2>{acc:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'xgb_mood' in models['performance']:
            acc = models['performance']['xgb_mood']['accuracy']
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Mood Model</h3>
                <h2>{acc:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Dataset overview with tabs
    st.markdown("### 📋 Dataset Overview")
    tab1, tab2, tab3 = st.tabs(["📊 Sample Data", "📈 Statistics", "🔍 Quick Insights"])
    
    with tab1:
        st.dataframe(data.head(10), use_container_width=True)
    
    with tab2:
        st.dataframe(data.describe(), use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 Key Findings:**
            - Screen time correlates with stress levels
            - Sleep patterns significantly impact mood scores
            - Social media usage affects mental well-being
            - Digital wellness scores predict mental health outcomes
            """)
        
        with col2:
            st.markdown("""
            **🎯 Model Performance:**
            - Random Forest: Stress prediction
            - XGBoost: Mood severity classification
            - K-Means: Lifestyle clustering
            - Isolation Forest: Anomaly detection
            """)

def show_predictions_page(models):
    st.markdown("## 🔮 Mental Health Predictions")
    st.markdown("Enter your digital habits to get personalized mental health insights.")
    
    # User input form with improved styling
    with st.form("prediction_form"):
        st.markdown("### 📝 Your Digital Habits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📱 Digital Usage**")
            screen_time = st.slider("Screen Time (hours/day)", 0.0, 24.0, 6.0, 0.1,
                                   help="Total time spent on screens including phone, computer, tablet")
            social_media = st.slider("Social Media Platforms Used", 0, 10, 3, 1,
                                   help="Number of different social media platforms you use regularly")
            reels_time = st.slider("Hours on Reels/Short Videos", 0.0, 10.0, 2.0, 0.1,
                                 help="Time spent on short-form video content")
        
        with col2:
            st.markdown("**😴 Sleep & Wellness**")
            sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.1,
                                  help="Average hours of sleep per night")
            stress_level = st.slider("Current Stress Level (1-10)", 1, 10, 5, 1,
                                   help="Rate your current stress level from 1 (low) to 10 (high)")
            mood_score = st.slider("Current Mood Score (1-10)", 1, 10, 7, 1,
                                 help="Rate your current mood from 1 (poor) to 10 (excellent)")
        
        submitted = st.form_submit_button("🔮 Get Predictions")
    
    if submitted:
        user_input = {
            'screen_time_hours': screen_time,
            'social_media_platforms_used': social_media,
            'hours_on_Reels': reels_time,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'mood_score': mood_score
        }
        
        predictions = predict_mental_health(user_input, models)
        
        st.markdown("## 📊 Your Mental Health Analysis")
        
        # Digital Wellness Gauge
        wellness_score = predictions['features']['digital_wellness_score']
        st.plotly_chart(create_digital_wellness_gauge(wellness_score), use_container_width=True)
        
        # Predictions in cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="prediction-box">
                <h3>🧠 Stress Level Prediction</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if predictions['stress_prediction'] == 1:
                st.error("⚠️ **High Stress Detected**")
                st.write("Your digital habits suggest high stress levels.")
            else:
                st.success("✅ **Low Stress Detected**")
                st.write("Your digital habits suggest manageable stress levels.")
            st.write(f"Confidence: {max(predictions['stress_probability']):.1%}")
        
        with col2:
            st.markdown("""
            <div class="prediction-box">
                <h3>😊 Mood Prediction</h3>
            </div>
            """, unsafe_allow_html=True)
            
            mood_labels = {0: "Low", 1: "Medium", 2: "High"}
            mood_pred = mood_labels.get(predictions['mood_prediction'], "Unknown")
            st.write(f"**{mood_pred} Mood**")
            st.write(f"Confidence: {max(predictions['mood_probability']):.1%}")
        
        # Lifestyle cluster
        st.markdown("""
        <div class="cluster-box">
            <h3>🎯 Lifestyle Cluster Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        cluster_descriptions = {
            0: "Balanced Digital Lifestyle - Moderate screen time with good sleep patterns",
            1: "High Digital Engagement - Extensive social media use with potential sleep impact",
            2: "Digital Wellness Focus - Low screen time with healthy sleep habits",
            3: "Stress-Prone Pattern - High screen time with poor sleep quality"
        }
        cluster = predictions['cluster']
        st.write(f"**Cluster {cluster}**: {cluster_descriptions.get(cluster, 'Unknown pattern')}")
        
        # Anomaly detection
        st.markdown("""
        <div class="anomaly-box">
            <h3>🔍 Anomaly Detection</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if predictions['is_anomaly']:
            st.warning("⚠️ **Anomalous Pattern Detected**")
            st.write("Your digital habits show unusual patterns compared to the general population.")
        else:
            st.success("✅ **Normal Pattern**")
            st.write("Your digital habits are within normal ranges.")
        st.write(f"Anomaly Score: {predictions['anomaly_score']:.3f}")
        
        # Recommendations
        st.markdown("## 💡 Personalized Recommendations")
        recommendations = []
        
        if screen_time > 8:
            recommendations.append("📱 Consider reducing screen time to improve sleep quality")
        if sleep_hours < 7:
            recommendations.append("😴 Aim for 7-9 hours of sleep for better mental health")
        if social_media > 5:
            recommendations.append("📱 Limit social media platforms to reduce digital overwhelm")
        if reels_time > 3:
            recommendations.append("⏰ Set time limits for short-form video consumption")
        if predictions['stress_prediction'] == 1:
            recommendations.append("🧘 Practice stress-reduction techniques like meditation")
        if predictions['mood_prediction'] == 0:
            recommendations.append("🌞 Increase exposure to natural light and physical activity")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation-card">
                    <p>{rec}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 Great job! Your digital habits are well-balanced.")

def show_analysis_page(data, models):
    st.markdown("## 📈 Data Analysis")
    
    # Model performance
    st.markdown("### 🎯 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'rf_stress' in models['performance']:
            perf = models['performance']['rf_stress']
            st.metric("Random Forest Accuracy", f"{perf['accuracy']:.2%}")
    
    with col2:
        if 'xgb_mood' in models['performance']:
            perf = models['performance']['xgb_mood']
            st.metric("XGBoost Accuracy", f"{perf['accuracy']:.2%}")
    
    # Feature importance
    st.markdown("### 🔍 Feature Importance")
    
    if 'rf_stress' in models['performance']:
        rf_importance = models['performance']['rf_stress']['feature_importance']
        st.plotly_chart(create_feature_importance_chart(rf_importance), use_container_width=True)

def show_data_explorer_page(data):
    st.markdown("## 📊 Data Explorer")
    
    # Correlation heatmap
    st.markdown("### 🔗 Feature Correlations")
    corr_matrix = data.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap of Features",
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive scatter plots
    st.markdown("### 📈 Interactive Scatter Plots")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("X-axis feature:", data.columns.tolist(), key='x')
    
    with col2:
        y_feature = st.selectbox("Y-axis feature:", data.columns.tolist(), key='y')
    
    if x_feature != y_feature:
        fig = px.scatter(
            data,
            x=x_feature,
            y=y_feature,
            title=f"{x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()}",
            opacity=0.6,
            color=y_feature,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.markdown("### 📊 Feature Distributions")
    selected_feature = st.selectbox("Select a feature to visualize:", data.columns.tolist())
    
    fig = px.histogram(
        data, 
        x=selected_feature,
        title=f"Distribution of {selected_feature.replace('_', ' ').title()}",
        nbins=30,
        color_discrete_sequence=['#667eea']
    )
    st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This application analyzes the relationship between digital habits and mental health outcomes.
    Using machine learning techniques, we can predict stress levels and mood severity based on
    digital behavior patterns.
    
    ### 🔬 Methodology
    
    **Machine Learning Models:**
    - **Random Forest**: Binary classification for stress prediction
    - **XGBoost**: Multi-class classification for mood severity
    - **K-Means Clustering**: Lifestyle pattern identification
    - **Isolation Forest**: Anomaly detection
    
    **Features Used:**
    - Screen time hours
    - Social media platforms used
    - Hours on Reels/short videos
    - Sleep hours
    - Derived features (digital wellness score, sleep quality, etc.)
    
    ### 🛠️ Technical Stack
    
    - **Python**: Core programming language
    - **Pandas**: Data manipulation and analysis
    - **Scikit-learn**: Machine learning algorithms
    - **XGBoost**: Gradient boosting for classification
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    
    ### 📈 Key Insights
    
    1. **Digital habits significantly impact mental health outcomes**
    2. **Screen time and sleep patterns are strong predictors of stress levels**
    3. **Social media usage patterns correlate with mood scores**
    4. **Lifestyle clustering reveals distinct behavioral patterns**
    5. **Anomaly detection helps identify unusual digital behavior patterns**
    
    ### 🚀 Getting Started
    
    1. Run the setup: `python simple_setup.py`
    2. Start the web app: `streamlit run app.py`
    3. Enter your digital habits to get personalized insights
    
    ### 📝 Disclaimer
    
    This application is for educational and research purposes only.
    It should not be used as a substitute for professional medical advice.
    If you're experiencing mental health concerns, please consult with a healthcare professional.
    """)

if __name__ == "__main__":
    main()
