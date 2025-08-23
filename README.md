# 🧠 Digital Habits vs Mental Health Analysis

A comprehensive machine learning project that analyzes the relationship between digital habits and mental health outcomes. This project includes data analysis, predictive modeling, clustering, anomaly detection, and an interactive web application.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project analyzes how digital habits (screen time, social media usage, sleep patterns) correlate with mental health indicators (stress levels, mood scores). The analysis includes:

- **Exploratory Data Analysis (EDA)**: Comprehensive data exploration and visualization
- **Supervised Learning**: Random Forest for stress prediction, XGBoost for mood classification
- **Unsupervised Learning**: K-Means clustering for lifestyle patterns, Isolation Forest for anomaly detection
- **Interactive Web App**: Streamlit application for real-time predictions and analysis

## ✨ Features

### 🔬 Data Analysis
- Comprehensive EDA with distribution plots and correlation analysis
- Feature engineering with derived metrics (digital wellness score, sleep quality, etc.)
- Missing value handling and data preprocessing

### 🤖 Machine Learning Models
- **Random Forest Classifier**: Binary classification for stress prediction
- **XGBoost Classifier**: Multi-class classification for mood severity
- **K-Means Clustering**: Lifestyle pattern identification
- **Isolation Forest**: Anomaly detection in digital behavior

### 📊 Visualizations
- Interactive plots using Plotly
- Correlation heatmaps
- Feature importance analysis
- Cluster visualizations
- Distribution plots

### 🌐 Web Application
- Interactive prediction interface
- Real-time mental health analysis
- Personalized recommendations
- Model performance metrics
- Data exploration tools

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**
   ```bash
   # Ensure you have the following files in your directory:
   # - digital_habits_vs_mental_health.csv
   # - requirements.txt
   # - All Python files
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import pandas, numpy, sklearn, xgboost, streamlit, plotly; print('All packages installed successfully!')"
   ```

## 📖 Usage

### 1. Quick Setup (Recommended)

```bash
python simple_setup.py
```

This will:
- Load and preprocess the dataset
- Train essential machine learning models
- Save models for the web app

### 2. Start the Interactive Web Application

```bash
streamlit run app.py
```

This will:
- Launch the Streamlit web app in your browser
- Provide interactive prediction interface
- Show data visualizations and analysis

### 3. Individual Module Usage

You can also run individual components:

```bash
# Run EDA only
python eda.py

# Run preprocessing only
python preprocessing.py

# Run model training only
python train_models.py
```

## 📁 Project Structure

```
MLPROJECTFSP/
├── digital_habits_vs_mental_health.csv    # Original dataset
├── requirements.txt                       # Python dependencies
├── README.md                             # Project documentation
├── INSTRUCTIONS.md                       # Quick start guide
├── app.py                                # Main Streamlit application
├── simple_setup.py                       # Quick model setup script
├── src/                                  # Source code modules
│   ├── eda.py                           # Data analysis
│   ├── preprocessing.py                 # Data preprocessing
│   ├── train_models.py                  # Model training
│   ├── main_analysis.py                 # Complete analysis pipeline
│   └── app.py                           # Original app (archived)
├── models/                               # Trained models (generated)
│   ├── rf_stress.joblib
│   ├── xgb_mood.joblib
│   ├── kmeans.joblib
│   ├── isolation_forest.joblib
│   ├── preprocessing.joblib
│   └── model_performance.joblib
├── plots/                                # Generated visualizations
└── reports/                              # Analysis reports
```

## 🔧 Technical Details

### Dataset Information
- **Source**: Digital habits vs mental health dataset
- **Records**: 100,000 observations
- **Features**: 6 original features + engineered features
- **Target Variables**: Stress level (binary), Mood severity (multi-class)

### Features Used
**Original Features:**
- `screen_time_hours`: Daily screen time in hours
- `social_media_platforms_used`: Number of social media platforms
- `hours_on_Reels`: Time spent on short-form videos
- `sleep_hours`: Daily sleep duration
- `stress_level`: Self-reported stress level (1-10)
- `mood_score`: Self-reported mood score (1-10)

**Engineered Features:**
- `digital_wellness_score`: Composite digital health metric
- `sleep_quality`: Binary indicator of good sleep (7-9 hours)
- `screen_sleep_ratio`: Ratio of screen time to sleep time
- `social_media_intensity`: Product of platforms and reels time
- `stress_mood_imbalance`: Absolute difference between stress and mood

### Machine Learning Models

#### Supervised Learning
1. **Random Forest Classifier**
   - **Target**: High stress prediction (binary)
   - **Features**: All engineered features
   - **Performance**: Accuracy ~85-90%

2. **XGBoost Classifier**
   - **Target**: Mood severity classification (3 classes)
   - **Features**: All engineered features
   - **Performance**: Accuracy ~80-85%

#### Unsupervised Learning
1. **K-Means Clustering**
   - **Purpose**: Lifestyle pattern identification
   - **Features**: Lifestyle-related features
   - **Clusters**: 4 optimal clusters identified

2. **Isolation Forest**
   - **Purpose**: Anomaly detection
   - **Features**: Lifestyle features
   - **Output**: Anomaly scores and labels

### Model Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Cross-validation**: 5-fold cross-validation scores
- **Feature Importance**: Model-specific feature rankings
- **Silhouette Score**: Clustering quality metric

## 📈 Results

### Key Findings
1. **Digital habits significantly impact mental health outcomes**
2. **Screen time and sleep patterns are strong predictors of stress levels**
3. **Social media usage patterns correlate with mood scores**
4. **Lifestyle clustering reveals distinct behavioral patterns**
5. **Anomaly detection helps identify unusual digital behavior patterns**

### Model Performance
- **Random Forest (Stress)**: ~85-90% accuracy
- **XGBoost (Mood)**: ~80-85% accuracy
- **K-Means Clustering**: Good silhouette scores
- **Anomaly Detection**: Identifies ~10% of data as anomalous

### Visualizations Generated
- Distribution plots for all features
- Correlation heatmap
- Stress-mood relationship analysis
- Digital habits analysis
- Cluster visualizations
- Feature importance plots
- Interactive plots (HTML format)

## 🎮 Web Application Features

### Home Page
- Dataset overview and key metrics
- Quick insights and model performance
- Recent activity status

### Predictions Page
- Interactive form for user input
- Real-time mental health predictions
- Personalized recommendations
- Cluster and anomaly analysis

### Analysis Page
- Model performance metrics
- Feature importance analysis
- Cross-validation results

### Visualizations Page
- Interactive correlation heatmap
- Feature distribution plots
- Customizable scatter plots

### About Page
- Project overview and methodology
- Technical stack information
- Getting started guide

## 🔍 Troubleshooting

### Common Issues

1. **Model files not found**
   ```
   Error: Model files not found! Please run the analysis pipeline first.
   Solution: Run `python main_analysis.py` before starting the web app
   ```

2. **Package installation issues**
   ```
   Error: ModuleNotFoundError
   Solution: Install requirements with `pip install -r requirements.txt`
   ```

3. **Memory issues with large dataset**
   ```
   Error: MemoryError
   Solution: Reduce dataset size or use data sampling
   ```

4. **Streamlit app not loading**
   ```
   Error: Streamlit connection issues
   Solution: Check if port 8501 is available, or use `streamlit run app.py --server.port 8502`
   ```

### Performance Tips
- Use a machine with at least 8GB RAM for optimal performance
- Close other applications when running the analysis pipeline
- For faster web app loading, ensure models are pre-trained

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is for educational and research purposes. Please ensure you have appropriate permissions for any data used.

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice. If you're experiencing mental health concerns, please consult with a healthcare professional.

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the error messages carefully
3. Ensure all dependencies are installed correctly
4. Verify the dataset file is present and accessible

---

**Happy Analyzing! 🧠📊**
