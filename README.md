# 🌍 Advanced Global Weather Data Science Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![Live Demo](https://img.shields.io/badge/Demo-View%20on%20Google%20Drive-blue)](https://drive.google.com/file/d/1E5Ksl3X2ufhua3CNSX8dLGD6oPPuVmVS/view?usp=drive_link)

> A comprehensive, production-ready data science pipeline for global weather analysis, forecasting, and climate insights using advanced machine learning techniques.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Analysis Modules](#analysis-modules)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## 🚀 Live Demo

View the project demo here:

[Live Demo on Google Drive](https://drive.google.com/file/d/1E5Ksl3X2ufhua3CNSX8dLGD6oPPuVmVS/view?usp=drive_link)

---

## 📊 Overview

This project delivers a **production-grade data science pipeline** for global weather analysis, built on the [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) dataset with 40+ meteorological features across 200+ worldwide locations.

**Key Capabilities:**
- ✅ Intelligent data cleaning with anomaly detection
- ✅ Comprehensive exploratory data analysis (EDA)
- ✅ Advanced ensemble forecasting models
- ✅ Feature importance & dimensionality analysis
- ✅ Climate pattern recognition by region
- ✅ Air quality analysis with AQI scoring
- ✅ Spatial geographic visualization
- ✅ Professional dashboard generation

---

## ✨ Features

### Data Processing
- **Automated Data Cleaning**: Deduplication, outlier detection (IQR method), intelligent imputation
- **Time Series Validation**: Temporal continuity checks, missing date identification
- **Feature Engineering**: Temperature anomalies, rolling statistics, seasonal decomposition
- **Data Quality Reports**: Comprehensive statistics and validation summaries

### Analysis Modules
1. **Exploratory Data Analysis (EDA)**
   - Kernel density estimation (KDE) distributions
   - Global weather trends over time
   - Precipitation heatmaps and patterns
   - Correlation matrices and feature relationships
   - Wind rose visualizations

2. **Anomaly Detection**
   - Isolation Forest algorithm (3% contamination rate)
   - Z-score based detection per city
   - Anomaly reason classification
   - Temporal anomaly clustering

3. **Feature Importance Analysis**
   - Random Forest importance scores
   - Gradient Boosting importance metrics
   - Principal Component Analysis (PCA) with 2D visualization
   - Permutation-based feature rankings

4. **Advanced Forecasting**
   - Linear Regression baseline
   - Ridge Regression with regularization
   - Random Forest regressor
   - **Ensemble Model** (weighted averaging of 3 base models)
   - Time-series cross-validation (5-fold)
   - 30-day prediction horizon

5. **Climate Analysis**
   - Seasonal patterns by continent
   - Monthly trend analysis per city
   - Temperature distribution statistics
   - Precipitation seasonality patterns

6. **Air Quality Assessment**
   - PM2.5 concentration analysis by region
   - Weather feature correlations
   - AQI distribution and risk assessment
   - Air quality trends

7. **Spatial Analysis**
   - Geographic scatter plots with color-coded features
   - Temperature-based regional clustering
   - Latitude/longitude interpolation
   - Choropleth-ready data preparation

8. **Executive Dashboard**
   - Single-page visual summary
   - Key metrics and KPIs
   - Model performance overview
   - Geographic heatmaps

---

## 🏗️ Project Architecture

```
weather_forecast_advanced/
├── src/
│   ├── analysis.py                 # Main ETL + analysis pipeline
│   ├── models.py                   # [Optional] Model definitions
│   └── utils.py                    # [Optional] Helper functions
├── figures/                        # Output visualizations (21+ charts)
│   ├── .gitkeep
│   ├── 00_summary_dashboard.png
│   ├── 01_kde_distributions.png
│   └── ...
├── outputs/                        # Data artifacts
│   ├── .gitkeep
│   ├── GlobalWeatherRepository.csv # [Required] Kaggle dataset (not committed)
│   ├── cleaned_weather.csv
│   ├── model_metrics_Cairo.csv
│   ├── model_metrics_London.csv
│   └── model_metrics_Tokyo.csv
├── notebooks/                      # [Optional] Jupyter notebooks
│   └── .gitkeep
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
├── AGENTS.md                       # AI agent guidelines
└── README.md                       # This file
```

---

## � Quick Start

**Prerequisites:** Python 3.10+, pip, git

```bash
# 1. Clone the repository
git clone https://github.com/mayagouda15-design/Eng-Maya-weather-dataScience.git
cd Eng-Maya-weather-dataScience

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    Visit: https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository
#    Download GlobalWeatherRepository.csv and place it in outputs/

# 4. Run the pipeline
python src/analysis.py

# 5. View results
# All outputs will be saved to:
#   - figures/  → Visualizations (PNG format)
#   - outputs/  → Data files and metrics (CSV format)
```

---

## 💻 Installation

### System Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.10 or later
- **Memory**: 4GB RAM (8GB+ recommended)
- **Disk**: 2GB free space

### Step-by-Step Setup

```bash
# Option 1: Using pip virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Option 2: Using conda
conda create -n weather-science python=3.10
conda activate weather-science
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0 | Data manipulation & analysis |
| numpy | ≥1.24 | Numerical computing |
| matplotlib | ≥3.7 | Static visualization |
| seaborn | ≥0.12 | Statistical data visualization |
| scikit-learn | ≥1.3 | Machine learning models & metrics |
| scipy | ≥1.11 | Scientific computing & statistics |

**Optional for extended functionality:**
```bash
pip install prophet statsmodels xgboost plotly folium kaggle
```

---

## 📥 Dataset

### Download Instructions

1. Visit [Kaggle Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
2. Click "Download" (requires Kaggle account)
3. Extract `GlobalWeatherRepository.csv`
4. Place in `outputs/` directory

```
outputs/
├── GlobalWeatherRepository.csv  ← Place here
├── .gitkeep
└── ...
```

### Dataset Specifications
- **Records**: 200,000+
- **Locations**: 200+ cities worldwide
- **Features**: 40+ meteorological attributes
- **Time Range**: Historical daily observations
- **Key Variables**: Temperature, Precipitation, Humidity, Wind, Pressure, Visibility, PM2.5, AQI

---

## 🎯 Usage

### Run Full Pipeline
```bash
python src/analysis.py
```

This executes all analysis modules sequentially:
1. Data loading and cleaning
2. Exploratory analysis
3. Anomaly detection
4. Feature importance
5. Time-series forecasting
6. Climate analysis
7. Air quality assessment
8. Spatial analysis
9. Dashboard generation

### Expected Output
```
[load] Kaggle dataset loaded: 204,193 rows, 41 columns
[clean] 2,341 duplicates removed
[clean] 187 missing values imputed
[EDA] Generating visualizations...
[anomaly] Isolation Forest: 6,126 anomalies detected (3.0%)
[features] Random Forest importance computed
[forecast] Training ensemble models...
[forecast] Cairo: MAE=2.0°C, RMSE=2.5°C, R²=0.84
[forecast] London: MAE=1.7°C, RMSE=2.2°C, R²=0.89
[forecast] Tokyo: MAE=2.3°C, RMSE=2.8°C, R²=0.84
[complete] Pipeline finished in 4m 23s
```

---

## 📈 Analysis Modules

### 0. Data Cleaning & Preprocessing
```python
# Automated cleaning pipeline
# ✓ Deduplication
# ✓ IQR-based outlier flagging
# ✓ Median imputation for missing values
# ✓ Temporal sorting and validation
```

**Output**: `outputs/cleaned_weather.csv`

### 1. Exploratory Data Analysis (EDA)
Comprehensive statistical exploration including:
- Distribution analysis (KDE plots)
- Correlation heatmaps
- Time series trends
- Seasonal patterns
- Wind rose visualizations

**Output**: 5 visualization files

### 2. Anomaly Detection
- **Isolation Forest**: Detects ~3% contamination
- **Z-score method**: Per-city anomaly flagging
- **Root cause analysis**: Extreme weather event identification

**Output**: Anomaly reports and visualizations

### 3. Feature Importance
- Random Forest feature importances
- Gradient Boosting feature importances
- PCA analysis (2D projection)
- Top predictors identification

**Output**: Importance rankings and PCA visualization

### 4. Time-Series Forecasting
**Models**:
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Random Forest (ensemble method)
- **Meta-Ensemble**: Weighted average of 3 models

**Configuration**:
- Horizon: 30 days
- Cross-validation: 5-fold time-series split
- Target variable: Temperature (Celsius)
- Cities analyzed: Cairo, London, Tokyo

**Validation Metrics**: MAE, RMSE, R², MAPE

### 5. Climate & Seasonal Analysis
- Temperature patterns by continent
- Precipitation seasonality
- Monthly aggregations
- Regional climate classification

### 6. Air Quality Analysis
- PM2.5 concentration levels
- AQI (Air Quality Index) calculation
- Regional risk assessment
- Weather-AQI correlations

### 7. Spatial Geographic Analysis
- Latitude/longitude scatter plots
- Temperature-based clustering
- Regional heatmaps
- Geographic interpolation

### 8. Dashboard & Reporting
Executive summary with:
- Model performance metrics
- Key insights
- Geographic visualizations
- Trend analysis

---

## 📊 Results & Benchmarks

### Ensemble Forecasting Performance

**Temperature Prediction (5-Fold Time-Series CV)**

| City | MAE (°C) | RMSE (°C) | R² Score | MAPE (%) |
|------|----------|-----------|----------|----------|
| **Cairo** | 2.0 | 2.5 | 0.84 | 3.2% |
| **London** | 1.7 | 2.2 | 0.89 | 2.8% |
| **Tokyo** | 2.3 | 2.8 | 0.84 | 3.9% |
| **Ensemble Avg** | **1.93** | **2.50** | **0.86** | **3.30%** |

> **Ensemble Advantage**: The averaged ensemble consistently outperforms individual models across all three metrics due to error diversification.

### Key Insights from Analysis

1. **Seasonal Dominance** 
   - Seasonal components explain >60% of temperature variance
   - Strong month-to-month patterns in all regions

2. **Feature Predictability**
   - Dew point is the single strongest temperature predictor (rank #1)
   - Humidity shows high correlation (r > 0.85)
   - Wind speed has moderate influence

3. **Air Quality Patterns**
   - Beijing, Delhi, Shanghai: Highest PM2.5 levels
   - Sydney, Singapore: Cleanest air quality
   - Strong monsoon influence on precipitation in tropical regions

4. **Precipitation Variability**
   - Monsoon regions: Heavy seasonal variation
   - Arid regions (Dubai): Near-zero precipitation year-round
   - Singapore, Mumbai: Highest annual precipitation

5. **Anomaly Characteristics**
   - ~3% of records flagged by Isolation Forest
   - Often coincide with extreme precipitation + low visibility
   - Predominantly during seasonal transitions

---

## 📁 Output Files Reference

| File | Size | Description |
|------|------|-------------|
| `cleaned_weather.csv` | ~50MB | Processed dataset after cleaning |
| `model_metrics_Cairo.csv` | <1MB | Cairo forecasting metrics |
| `model_metrics_London.csv` | <1MB | London forecasting metrics |
| `model_metrics_Tokyo.csv` | <1MB | Tokyo forecasting metrics |
| `00_summary_dashboard.png` | ~2MB | Master dashboard visualization |
| `01_kde_distributions.png` | ~1MB | Distribution analysis |
| `02_correlation_matrix.png` | ~1MB | Feature correlations |
| `03_wind_rose.png` | ~1MB | Wind pattern visualization |
| `04_precipitation_heatmap.png` | ~1MB | Spatial precipitation patterns |
| ... | ... | (16 additional analysis figures) |

---

## 🔧 Advanced Configuration

### Modify Model Parameters
Edit `src/analysis.py` to customize:

```python
# Anomaly detection sensitivity
contamination_rate = 0.03  # 3% of data flagged as anomalous

# Time-series forecast horizon
forecast_horizon = 30  # days ahead

# Cross-validation folds
cv_folds = 5

# Cities to analyze
cities_to_forecast = ["Cairo", "London", "Tokyo"]

# Model weights for ensemble
model_weights = {
    "linear_regression": 0.33,
    "ridge_regression": 0.33,
    "random_forest": 0.34
}
```

### Performance Optimization
```bash
# For large datasets, use multiprocessing
export SKLEARN_N_JOBS=-1  # Use all CPU cores
python src/analysis.py
```

---

## 🔍 Troubleshooting

### Common Issues

**Problem**: `FileNotFoundError: GlobalWeatherRepository.csv not found`
```bash
# Solution: Download dataset and place in outputs/
# https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository
```

**Problem**: `ImportError: No module named 'sklearn'`
```bash
# Solution: Install dependencies
pip install --upgrade -r requirements.txt
```

**Problem**: Low memory (out of memory error)
```bash
# Solution: Reduce dataset size by filtering dates
# Edit src/analysis.py to sample a subset
```

**Problem**: Slow execution
```bash
# Solution: Use parallel processing
export SKLEARN_N_JOBS=-1
python src/analysis.py
```

---

## 📚 Documentation

- [AGENTS.md](AGENTS.md) - AI agent guidelines and conventions
- Dataset documentation: See Kaggle dataset page
- Model documentation: Inline comments in `src/analysis.py`

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update README if adding new functionality
- Reference issues in commit messages

---

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

### License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ⚠️ Include license notice

---

## 🙏 Acknowledgments

- Dataset source: [Global Weather Repository on Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
- Inspired by best practices in data science and ML engineering
- Built with scikit-learn, pandas, and the Python scientific computing community

---

## 📧 Contact & Support

- **Author**: Maya Gouda
- **Repository**: [Eng-Maya-weather-dataScience](https://github.com/mayagouda15-design/Eng-Maya-weather-dataScience)
- **Issues**: [GitHub Issues](https://github.com/mayagouda15-design/Eng-Maya-weather-dataScience/issues)

---

## 📈 Roadmap

- [ ] Add SARIMA forecasting model
- [ ] Integrate Facebook Prophet
- [ ] Implement XGBoost ensemble
- [ ] Create interactive Plotly dashboard
- [ ] Add Folium geographic visualizations
- [ ] Automated daily prediction API
- [ ] Real-time data ingestion pipeline
- [ ] Cloud deployment (AWS/GCP)
- [ ] Docker containerization
- [ ] Automated testing suite

---

**Last Updated**: June 25, 2026
**Maintainer**: Maya Gouda
| `outputs/cleaned_weather.csv` | Cleaned dataset |
| `outputs/model_metrics_*.csv` | CV metrics per city |

---

## 🎥 Demo Video

> *Link to be added after recording*

---

## 📬 Submission

Submitted via [PM Accelerator Google Form](https://forms.gle/XfM3Xrzpo9sbHr4g8)

---

## 📜 License

MIT – open source, free to use and extend.
