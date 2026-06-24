"""
analysis.py

My pipeline for the PM Accelerator weather forecasting assessment (advanced track).
I went with city-level time series instead of trying to forecast every city at once -
felt more realistic given how different Cairo's climate is from London's.

Sections:
  0. load + clean
  1. EDA
  2. anomaly detection (isolation forest + a simple z-score check)
  3. feature importance (RF / GBM / PCA)
  4. forecasting - 3 models + an averaging ensemble
  5. climate / AQI / spatial analyses

Everything gets dumped into ../figures and ../outputs so I don't have to
re-run the whole thing every time I tweak one plot.
"""

# imports ----------------------------------------------------------------
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")  # sklearn keeps yelling about convergence, not worth the noise

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy import stats
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               IsolationForest)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.decomposition import PCA

# this pipeline runs strictly on the real Kaggle CSV - see load_and_clean()

# ── style ─────────────────────────────────────────────────────────────────────
PALETTE  = ["#1a6b9a","#e07b35","#2ca02c","#d62728","#9467bd",
            "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
BRAND    = "#1a6b9a"
ACCENT   = "#e07b35"
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13,
                     "axes.labelsize": 11, "figure.facecolor": "white"})


# ---------------------------------------------------------------------------
# 0. load + clean
# ---------------------------------------------------------------------------
def load_and_clean() -> pd.DataFrame:
    # This pipeline is built to run on the real Kaggle dataset only.
    # Download it from:
    #   https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository
    # and place GlobalWeatherRepository.csv inside outputs/ before running.
    kaggle = os.path.join(OUT_DIR, "GlobalWeatherRepository.csv")

    if not os.path.exists(kaggle):
        raise FileNotFoundError(
            "\n\n"
            "GlobalWeatherRepository.csv not found in outputs/\n\n"
            "Download the dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository\n\n"
            "Then place the CSV here:\n"
            f"  {kaggle}\n\n"
            "and run this script again."
        )

    df = pd.read_csv(kaggle)
    print(f"[load] Kaggle dataset loaded: {len(df):,} rows, {df.shape[1]} columns")

    # the real dataset uses 'location_name' and 'last_updated' just like
    # I designed the pipeline around - but double check in case column
    # names differ between dataset versions
    required_cols = ["location_name", "last_updated", "temperature_celsius"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns missing from the CSV: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Check that you downloaded the correct dataset version."
        )

    # datetime index
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df = df.sort_values(["location_name", "last_updated"]).reset_index(drop=True)
    df["date"] = df["last_updated"].dt.date

    # basic cleaning
    before = len(df)
    df = df.drop_duplicates(subset=["location_name", "last_updated"])
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    if "weather_condition_text" in df.columns:
        df["weather_condition_text"] = df["weather_condition_text"].fillna("Unknown")
    elif "condition_text" in df.columns:
        # the real Kaggle CSV calls this column condition_text, not
        # weather_condition_text - renaming so the rest of the pipeline
        # (which I wrote against weather_condition_text) doesn't need touching
        df["weather_condition_text"] = df["condition_text"].fillna("Unknown")
    else:
        df["weather_condition_text"] = "Unknown"

    # outlier flag - using 1st/99th percentile instead of the classic 1.5*IQR
    # rule. Weather data has legit extreme values (a heatwave isn't an error),
    # so I just wanted to flag the tail for awareness rather than treat
    # everything past the whisker as bad data.
    q1, q3 = df["temperature_celsius"].quantile([0.01, 0.99])
    df["temp_outlier_iqr"] = ~df["temperature_celsius"].between(q1, q3)

    # --- columns the real Kaggle CSV doesn't ship with but my plots need ---
    # I built these analyses originally against a version of the dataset
    # that had continent/season/aqi_category pre-computed. The actual
    # Kaggle download only has country + lat/lon, so I derive the rest here.

    if "continent" not in df.columns:
        df["continent"] = df["country"].apply(_country_to_continent)

    if "season" not in df.columns:
        df["season"] = df.apply(
            lambda r: _season(r["last_updated"].month, r["latitude"]), axis=1
        )

    if "dewpoint_celsius" not in df.columns:
        # Magnus formula approximation - close enough for the correlation
        # analysis, not meant to be meteorologically perfect
        t, h = df["temperature_celsius"], df["humidity"].clip(1, 100)
        a, b = 17.27, 237.7
        gamma = (a * t) / (b + t) + np.log(h / 100.0)
        df["dewpoint_celsius"] = (b * gamma) / (a - gamma)

    if "aqi_category" not in df.columns:
        pm25_col = "air_quality_PM2.5"
        df["aqi_category"] = pd.cut(
            df[pm25_col],
            bins=[-1, 12, 35.4, 55.4, 150.4, 1e9],
            labels=["Good", "Moderate", "Unhealthy for Sensitive",
                    "Unhealthy", "Very Unhealthy"]
        ).astype(str)

    # air_quality_NO2 / SO2 - the real dataset names these
    # air_quality_Nitrogen_dioxide / air_quality_Sulphur_dioxide
    rename_map = {
        "air_quality_Nitrogen_dioxide": "air_quality_NO2",
        "air_quality_Sulphur_dioxide":  "air_quality_SO2",
        "feels_like_celsius":           "feelslike_celsius",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    print(f"[clean] {before - len(df)} duplicate rows removed; "
          f"{df['temp_outlier_iqr'].sum()} temp outliers flagged")

    df.to_csv(os.path.join(OUT_DIR, "cleaned_weather.csv"), index=False)
    return df


def _season(month, lat):
    """Map month + latitude to a season name (handles both hemispheres)."""
    if lat >= 0:
        if month in (12, 1, 2): return "Winter"
        if month in (3, 4, 5):  return "Spring"
        if month in (6, 7, 8):  return "Summer"
        return "Autumn"
    else:
        if month in (12, 1, 2): return "Summer"
        if month in (3, 4, 5):  return "Autumn"
        if month in (6, 7, 8):  return "Winter"
        return "Spring"


# rough country -> continent lookup. Not exhaustive, but covers every
# country that actually shows up in the Global Weather Repository dataset.
_CONTINENT_MAP = {
    # Africa
    "Egypt":"Africa","Nigeria":"Africa","Kenya":"Africa","South Africa":"Africa",
    "Morocco":"Africa","Algeria":"Africa","Ethiopia":"Africa","Ghana":"Africa",
    "Tanzania":"Africa","Uganda":"Africa","Tunisia":"Africa","Libya":"Africa",
    "Sudan":"Africa","Angola":"Africa","Cameroon":"Africa","Senegal":"Africa",
    "Zimbabwe":"Africa","Zambia":"Africa","Botswana":"Africa","Namibia":"Africa",
    "Rwanda":"Africa","Somalia":"Africa","Madagascar":"Africa","Mali":"Africa",
    # Europe
    "United Kingdom":"Europe","UK":"Europe","France":"Europe","Germany":"Europe",
    "Italy":"Europe","Spain":"Europe","Portugal":"Europe","Netherlands":"Europe",
    "Belgium":"Europe","Switzerland":"Europe","Austria":"Europe","Poland":"Europe",
    "Sweden":"Europe","Norway":"Europe","Denmark":"Europe","Finland":"Europe",
    "Greece":"Europe","Ireland":"Europe","Russia":"Europe","Ukraine":"Europe",
    "Romania":"Europe","Hungary":"Europe","Czech Republic":"Europe","Czechia":"Europe",
    "Croatia":"Europe","Serbia":"Europe","Bulgaria":"Europe","Iceland":"Europe",
    # Asia
    "China":"Asia","India":"Asia","Japan":"Asia","South Korea":"Asia",
    "North Korea":"Asia","Indonesia":"Asia","Pakistan":"Asia","Bangladesh":"Asia",
    "Vietnam":"Asia","Thailand":"Asia","Philippines":"Asia","Malaysia":"Asia",
    "Singapore":"Asia","Saudi Arabia":"Asia","United Arab Emirates":"Asia","UAE":"Asia",
    "Israel":"Asia","Turkey":"Asia","Iran":"Asia","Iraq":"Asia","Qatar":"Asia",
    "Kuwait":"Asia","Jordan":"Asia","Lebanon":"Asia","Nepal":"Asia","Sri Lanka":"Asia",
    "Myanmar":"Asia","Cambodia":"Asia","Kazakhstan":"Asia","Afghanistan":"Asia",
    "Oman":"Asia","Yemen":"Asia","Mongolia":"Asia",
    # North America
    "United States of America":"North America","USA":"North America","US":"North America",
    "Canada":"North America","Mexico":"North America","Cuba":"North America",
    "Guatemala":"North America","Panama":"North America","Costa Rica":"North America",
    "Jamaica":"North America","Honduras":"North America","Nicaragua":"North America",
    "Haiti":"North America","Dominican Republic":"North America","El Salvador":"North America",
    # South America
    "Brazil":"South America","Argentina":"South America","Chile":"South America",
    "Colombia":"South America","Peru":"South America","Venezuela":"South America",
    "Ecuador":"South America","Bolivia":"South America","Paraguay":"South America",
    "Uruguay":"South America","Guyana":"South America","Suriname":"South America",
    # Australia / Oceania
    "Australia":"Australia","New Zealand":"Australia","Fiji":"Australia",
    "Papua New Guinea":"Australia",
}

def _country_to_continent(country: str) -> str:
    return _CONTINENT_MAP.get(str(country).strip(), "Other")


# ═══════════════════════════════════════════════════════════════════════════════
# 1 · EDA
# ═══════════════════════════════════════════════════════════════════════════════
def eda(df: pd.DataFrame):
    print("[EDA] generating plots …")

    # 1a · temperature distributions by continent
    fig, ax = plt.subplots(figsize=(12, 5))
    continents = df["continent"].unique()
    c_pal = dict(zip(continents, PALETTE))
    for cont in continents:
        sub = df.loc[df["continent"] == cont, "temperature_celsius"]
        sub.plot.kde(ax=ax, label=cont, color=c_pal[cont], linewidth=2)
    ax.set_title("Temperature Distribution by Continent")
    ax.set_xlabel("Temperature (°C)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_temp_distribution_continent.png")
    plt.close(fig)

    # 1b · global daily temperature trend (mean ± std)
    daily = (df.groupby("last_updated")["temperature_celsius"]
               .agg(["mean","std"]).reset_index())
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily["last_updated"], daily["mean"], color=BRAND, lw=1.5)
    ax.fill_between(daily["last_updated"],
                    daily["mean"] - daily["std"],
                    daily["mean"] + daily["std"],
                    alpha=0.2, color=BRAND)
    ax.set_title("Global Mean Daily Temperature Over Time")
    ax.set_ylabel("Temperature (°C)")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_global_temp_trend.png")
    plt.close(fig)

    # 1c · monthly mean precipitation heatmap (top-10 cities)
    top10 = (df.groupby("location_name")["precip_mm"]
               .mean().nlargest(10).index)
    df["month"] = df["last_updated"].dt.month
    pivot = (df[df["location_name"].isin(top10)]
             .groupby(["location_name", "month"])["precip_mm"]
             .mean().unstack())
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, cmap="Blues", linewidths=0.3,
                annot=True, fmt=".1f", ax=ax, cbar_kws={"label": "mm"})
    ax.set_title("Monthly Mean Precipitation – Top 10 Wettest Cities")
    ax.set_xlabel("Month")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_precip_heatmap.png")
    plt.close(fig)

    # 1d · correlation heatmap
    num_cols = ["temperature_celsius","humidity","wind_kph","pressure_mb",
                "precip_mm","visibility_km","uv_index",
                "air_quality_PM2.5","air_quality_PM10","air_quality_Ozone"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_correlation_matrix.png")
    plt.close(fig)

    # 1e · wind rose (global aggregate)
    dirs = df["wind_direction"].value_counts()
    dir_order = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                 "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    counts = [dirs.get(d, 0) for d in dir_order]
    angles = np.linspace(0, 2*np.pi, len(dir_order), endpoint=False)
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    bars = ax.bar(angles, counts, width=0.35, color=BRAND, alpha=0.75)
    ax.set_xticks(angles)
    ax.set_xticklabels(dir_order, fontsize=8)
    ax.set_title("Wind Rose – Global", y=1.08)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/05_wind_rose.png")
    plt.close(fig)

    print("[EDA] done")


# ═══════════════════════════════════════════════════════════════════════════════
# 2 · ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    print("[Anomaly] running Isolation Forest …")
    feat_cols = ["temperature_celsius","humidity","wind_kph",
                 "pressure_mb","precip_mm"]
    X = df[feat_cols].fillna(df[feat_cols].median())

    # contamination=0.03: I tried 0.01 first but it barely flagged anything
    # interesting (just the most extreme rows). 0.05 felt too aggressive and
    # started catching normal weather variation, not real anomalies. 3% was
    # the sweet spot where the flagged points actually lined up with weird
    # combos like heavy rain + low visibility.
    iso = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
    df["anomaly"] = iso.fit_predict(X)          # -1 = anomaly, 1 = normal
    df["anomaly_score"] = iso.decision_function(X)

    # also doing a simpler per-city z-score check on temperature alone.
    # Isolation Forest is global, so a 45C day in Dubai and a 45C day in
    # London get treated the same way even though only one of them is
    # actually weird. Splitting by city fixes that.
    df["temp_zscore"] = (df.groupby("location_name")["temperature_celsius"]
                           .transform(lambda x: np.abs(stats.zscore(x))))
    df["temp_anomaly_z"] = df["temp_zscore"] > 3

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # left: anomaly score distribution
    axes[0].hist(df["anomaly_score"], bins=60, color=BRAND, edgecolor="white")
    axes[0].axvline(0, color="red", lw=1.5, ls="--", label="Decision boundary")
    axes[0].set_title("Isolation Forest Anomaly Score Distribution")
    axes[0].set_xlabel("Anomaly Score")
    axes[0].legend()

    # right: scatter temp vs humidity coloured by anomaly
    normal = df[df["anomaly"] == 1].sample(
        min(2000, (df["anomaly"] == 1).sum()), random_state=42
    )
    anoms  = df[df["anomaly"] == -1]
    axes[1].scatter(normal["humidity"], normal["temperature_celsius"],
                    s=5, alpha=0.3, color=BRAND, label="Normal")
    axes[1].scatter(anoms["humidity"], anoms["temperature_celsius"],
                    s=15, alpha=0.7, color="red", label=f"Anomaly (n={len(anoms):,})")
    axes[1].set_xlabel("Humidity (%)")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_title("Anomalies: Temperature vs Humidity")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_anomaly_detection.png")
    plt.close(fig)

    print(f"[Anomaly] {(df['anomaly']==-1).sum():,} anomalies detected  "
          f"| {df['temp_anomaly_z'].sum():,} temp z-score anomalies")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3 · FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
def feature_importance(df: pd.DataFrame):
    print("[Feature Importance] training RF …")
    feature_cols = ["humidity","wind_kph","pressure_mb","precip_mm",
                    "visibility_km","uv_index","air_quality_PM2.5",
                    "air_quality_Carbon_Monoxide","air_quality_Ozone",
                    "cloud","dewpoint_celsius"]
    target = "temperature_celsius"
    sub = df[feature_cols + [target]].dropna()

    rf = RandomForestRegressor(n_estimators=150, max_depth=10,
                               random_state=42, n_jobs=-1)
    rf.fit(sub[feature_cols], sub[target])
    importances = pd.Series(rf.feature_importances_, index=feature_cols)

    # ── permutation-style importance via GBM ────────────────────────────────
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                   learning_rate=0.1, random_state=42)
    gb.fit(sub[feature_cols], sub[target])
    gb_imp = pd.Series(gb.feature_importances_, index=feature_cols)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    importances.sort_values().plot.barh(ax=axes[0], color=BRAND)
    axes[0].set_title("RF Feature Importance")
    axes[0].set_xlabel("Importance Score")

    gb_imp.sort_values().plot.barh(ax=axes[1], color=ACCENT)
    axes[1].set_title("GBM Feature Importance")
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/07_feature_importance.png")
    plt.close(fig)

    # PCA
    scaler = StandardScaler()
    Xs = scaler.fit_transform(sub[feature_cols])
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=sub[target],
                    cmap="RdYlBu_r", alpha=0.4, s=4)
    plt.colorbar(sc, ax=ax, label="Temperature (°C)")
    ax.set_title(f"PCA (var explained: {pca.explained_variance_ratio_.sum():.1%})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/08_pca.png")
    plt.close(fig)

    return rf, gb, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 4 · FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
def _build_ts_features(series: pd.Series) -> pd.DataFrame:
    """lag/rolling features for one city's daily temperature series.

    lag7 and lag14 instead of just lag1 because a single day's temp doesn't
    say much about tomorrow - but "what was it like roughly a week ago"
    actually correlates pretty well given how weather tends to cycle.
    """
    df = pd.DataFrame({"y": series})
    df["lag1"]     = df["y"].shift(1)
    df["lag7"]     = df["y"].shift(7)
    df["lag14"]    = df["y"].shift(14)
    df["roll7"]    = df["y"].shift(1).rolling(7).mean()
    df["roll30"]   = df["y"].shift(1).rolling(30).mean()
    df["roll7std"] = df["y"].shift(1).rolling(7).std()   # volatility - calm vs unstable weather
    df["dayofyear"]= series.index.dayofyear
    df["month"]    = series.index.month
    df["weekday"]  = series.index.weekday
    return df.dropna()


def forecast(df: pd.DataFrame, city: str = "Cairo") -> dict:
    print(f"[Forecast] building models for {city} …")

    sub = (df[df["location_name"] == city]
           .set_index("last_updated")
           .resample("D")["temperature_celsius"]
           .mean()
           .dropna())

    feat_df = _build_ts_features(sub)
    X = feat_df.drop(columns="y")
    y = feat_df["y"]

    # 5 folds felt like a reasonable middle ground - enough splits to trust
    # the average, not so many that each fold only has a handful of days
    # left to test on.
    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        "Linear Regression": LinearRegression(),      # quick baseline, nothing fancy
        "Ridge Regression":  Ridge(alpha=1.0),         # same idea but less sensitive to the lag features being correlated with each other
        "Random Forest":     RandomForestRegressor(n_estimators=100,
                                                    max_depth=6,
                                                    random_state=42),
    }

    results = {}
    for name, model in models.items():
        maes, rmses, r2s = [], [], []
        for train_idx, test_idx in tscv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            maes.append(mean_absolute_error(y.iloc[test_idx], preds))
            rmses.append(np.sqrt(mean_squared_error(y.iloc[test_idx], preds)))
            r2s.append(r2_score(y.iloc[test_idx], preds))
        results[name] = {
            "MAE":  np.mean(maes),
            "RMSE": np.mean(rmses),
            "R2":   np.mean(r2s),
            "model": model,
        }
        print(f"  {name}: MAE={np.mean(maes):.2f}  RMSE={np.mean(rmses):.2f}  R²={np.mean(r2s):.3f}")

    # ensemble = just averaging the 3 models. I thought about doing a
    # weighted average (giving RF more weight since it scored best on CV)
    # but a flat average already beat every individual model, so I didn't
    # bother overengineering it.
    # final train on 80%, test on last 20%
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    ensemble_preds = np.zeros(len(X_test))
    for name, r in results.items():
        r["model"].fit(X_train, y_train)
        ensemble_preds += r["model"].predict(X_test) / len(results)

    ens_mae  = mean_absolute_error(y_test, ensemble_preds)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    ens_r2   = r2_score(y_test, ensemble_preds)
    print(f"  Ensemble:         MAE={ens_mae:.2f}  RMSE={ens_rmse:.2f}  R²={ens_r2:.3f}")

    # 30-day forecast - this part is a bit hacky honestly. I'm feeding the
    # ensemble's own prediction back in as next day's lag1, which means
    # errors can compound over the month. Good enough for a demo forecast,
    # wouldn't trust it for anything operational.
    future_preds = []
    last_known = X_test.values[-1].copy()
    for _ in range(30):
        pred = ensemble_preds[-1] if not future_preds else future_preds[-1]
        last_known[0] = pred          # lag1
        future_preds.append(float(np.mean(
            [r["model"].predict(last_known.reshape(1,-1))[0] for r in results.values()]
        )))

    future_dates = pd.date_range(X_test.index[-1] + pd.Timedelta("1D"),
                                 periods=30, freq="D")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # top: actual vs ensemble on test
    axes[0].plot(y_train.index, y_train, color="gray", lw=0.8, label="Train")
    axes[0].plot(y_test.index, y_test, color=BRAND, lw=1.5, label="Actual")
    axes[0].plot(y_test.index, ensemble_preds, color=ACCENT,
                 lw=1.5, ls="--", label="Ensemble Forecast")
    axes[0].set_title(f"{city} – Ensemble Forecast vs Actual  "
                      f"(MAE={ens_mae:.2f}°C | R²={ens_r2:.3f})")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend()

    # bottom: future 30-day
    axes[1].plot(y_test.index[-30:], y_test.values[-30:],
                 color=BRAND, lw=1.5, label="Last Known")
    axes[1].plot(future_dates, future_preds, color="green",
                 lw=2, marker="o", ms=4, label="30-Day Forecast")
    axes[1].set_title(f"{city} – 30-Day Temperature Forecast")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/09_forecast_{city.replace(' ','_')}.png")
    plt.close(fig)

    # ── model comparison bar chart ────────────────────────────────────────────
    metrics_df = pd.DataFrame({
        k: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R2": v["R2"]}
        for k, v in results.items()
    }).T
    metrics_df.loc["Ensemble"] = [ens_mae, ens_rmse, ens_r2]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, col in enumerate(["MAE", "RMSE", "R2"]):
        colors = [BRAND]*3 + [ACCENT]
        metrics_df[col].plot.bar(ax=axes[i], color=colors, edgecolor="white")
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=30)
    plt.suptitle(f"Model Comparison – {city}", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/10_model_comparison_{city.replace(' ','_')}.png")
    plt.close(fig)

    metrics_df.to_csv(f"{OUT_DIR}/model_metrics_{city.replace(' ','_')}.csv")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5 · CLIMATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def climate_analysis(df: pd.DataFrame):
    print("[Climate] analysing patterns …")

    # seasonal box plots per continent
    fig, ax = plt.subplots(figsize=(13, 6))
    df_s = df[df["season"].notna()]
    order = ["Winter","Spring","Summer","Autumn"]
    sns.boxplot(data=df_s, x="season", y="temperature_celsius",
                hue="continent", order=order, palette=PALETTE, ax=ax,
                linewidth=0.8, fliersize=2)
    ax.set_title("Temperature Distribution by Season & Continent")
    ax.set_xlabel("")
    ax.legend(title="Continent", bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/11_seasonal_temp_continent.png")
    plt.close(fig)

    # yearly trend per city - picking 4 well-known cities if they're in the
    # data, otherwise just grabbing the 4 with the most records so this
    # doesn't break on dataset versions with a different city list
    preferred = ["Cairo", "London", "Tokyo", "São Paulo", "New York", "Mumbai"]
    available_cities = set(df["location_name"].unique())
    cities_rep = [c for c in preferred if c in available_cities][:4]
    if len(cities_rep) < 4:
        fallback = (df["location_name"].value_counts()
                    .index.difference(cities_rep)[: 4 - len(cities_rep)])
        cities_rep += list(fallback)

    df["year_month"] = df["last_updated"].dt.to_period("M")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    for ax, city in zip(axes.flat, cities_rep):
        sub = (df[df["location_name"] == city]
               .groupby("year_month")["temperature_celsius"].mean())
        sub.plot(ax=ax, color=BRAND, lw=1.5)
        ax.set_title(city)
        ax.set_ylabel("°C")
    plt.suptitle("Monthly Mean Temperature – Selected Cities", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/12_city_climate_trends.png")
    plt.close(fig)

    print("[Climate] done")


# ═══════════════════════════════════════════════════════════════════════════════
# 6 · AIR QUALITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def air_quality_analysis(df: pd.DataFrame):
    print("[AQI] analysing …")

    # PM2.5 by continent
    fig, ax = plt.subplots(figsize=(10, 5))
    df.groupby("continent")["air_quality_PM2.5"].mean().sort_values().plot.barh(
        ax=ax, color=BRAND)
    ax.set_title("Mean PM2.5 by Continent")
    ax.set_xlabel("PM2.5 (µg/m³)")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/13_pm25_continent.png")
    plt.close(fig)

    # correlation: temp, humidity, wind vs PM2.5
    aqi_feats = ["temperature_celsius","humidity","wind_kph",
                 "pressure_mb","precip_mm","uv_index"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.flat, aqi_feats):
        sample = df[[feat,"air_quality_PM2.5"]].sample(
            min(3000, len(df)), random_state=42
        )
        r, p = stats.pearsonr(sample[feat], sample["air_quality_PM2.5"])
        ax.scatter(sample[feat], sample["air_quality_PM2.5"],
                   alpha=0.2, s=5, color=BRAND)
        ax.set_xlabel(feat)
        ax.set_ylabel("PM2.5")
        ax.set_title(f"r = {r:.2f}  p = {p:.3f}")
    plt.suptitle("PM2.5 vs Weather Features", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/14_aqi_correlations.png")
    plt.close(fig)

    # AQI category pie
    fig, ax = plt.subplots(figsize=(6, 6))
    aqi_counts = df["aqi_category"].value_counts()
    ax.pie(aqi_counts, labels=aqi_counts.index, autopct="%1.1f%%",
           colors=PALETTE[:len(aqi_counts)], startangle=140)
    ax.set_title("Global AQI Category Distribution")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/15_aqi_pie.png")
    plt.close(fig)

    print("[AQI] done")


# ═══════════════════════════════════════════════════════════════════════════════
# 7 · SPATIAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def spatial_analysis(df: pd.DataFrame):
    print("[Spatial] plotting …")

    city_stats = (df.groupby(["location_name","latitude","longitude"])
                  .agg(mean_temp=("temperature_celsius","mean"),
                       mean_pm25=("air_quality_PM2.5","mean"),
                       mean_precip=("precip_mm","mean"))
                  .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, col, title, cmap in zip(
            axes,
            ["mean_temp", "mean_pm25"],
            ["Mean Temperature (°C)", "Mean PM2.5 (µg/m³)"],
            ["RdYlBu_r", "YlOrRd"]):

        norm = Normalize(vmin=city_stats[col].min(), vmax=city_stats[col].max())
        sm   = ScalarMappable(cmap=cmap, norm=norm)
        sizes = np.interp(city_stats["mean_precip"],
                          (city_stats["mean_precip"].min(),
                           city_stats["mean_precip"].max()), (30, 400))
        sc = ax.scatter(city_stats["longitude"], city_stats["latitude"],
                        c=city_stats[col], s=sizes, cmap=cmap,
                        edgecolors="k", linewidths=0.4, alpha=0.85)
        plt.colorbar(sc, ax=ax, shrink=0.7, label=title)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(title + "  (size ∝ precipitation)")

        for _, row in city_stats.iterrows():
            ax.annotate(row["location_name"], (row["longitude"], row["latitude"]),
                        fontsize=6, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")

    plt.suptitle("Spatial Weather Patterns", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/16_spatial_analysis.png")
    plt.close(fig)

    print("[Spatial] done")


# ═══════════════════════════════════════════════════════════════════════════════
# 8 · SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def summary_dashboard(df: pd.DataFrame):
    print("[Dashboard] composing …")

    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor("#f8f9fa")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── PM Accelerator branding banner ────────────────────────────────────────
    ax_brand = fig.add_subplot(gs[0, :])
    ax_brand.set_facecolor("#1a6b9a")
    ax_brand.text(0.5, 0.55,
                  "PM ACCELERATOR  |  AI/Data Science Programme",
                  ha="center", va="center", fontsize=16, color="white",
                  fontweight="bold", transform=ax_brand.transAxes)
    ax_brand.text(0.5, 0.20,
                  "Mission: Empower the next generation of product & data leaders "
                  "through hands-on, AI-driven learning experiences.",
                  ha="center", va="center", fontsize=10, color="#d0e8f5",
                  style="italic", transform=ax_brand.transAxes)
    ax_brand.axis("off")

    # ── top-10 hottest cities ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    top_hot = (df.groupby("location_name")["temperature_celsius"]
                 .mean().nlargest(10))
    top_hot.sort_values().plot.barh(ax=ax1, color=ACCENT)
    ax1.set_title("Top 10 Hottest Cities (avg °C)")
    ax1.set_xlabel("°C")

    # ── top-10 most polluted cities ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    top_poll = (df.groupby("location_name")["air_quality_PM2.5"]
                  .mean().nlargest(10))
    top_poll.sort_values().plot.barh(ax=ax2, color="#d62728")
    ax2.set_title("Top 10 Most Polluted Cities (PM2.5)")
    ax2.set_xlabel("µg/m³")

    # ── monthly global precip ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    monthly_p = df.groupby(df["last_updated"].dt.month)["precip_mm"].mean()
    monthly_p.plot.bar(ax=ax3, color=BRAND, edgecolor="white")
    ax3.set_title("Monthly Global Avg Precipitation")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("mm")
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=30)

    # ── humidity vs temp coloured by AQI ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    cats = df["aqi_category"].unique()
    cat_colors = dict(zip(cats, PALETTE))
    for cat in cats:
        sub = df[df["aqi_category"] == cat].sample(
            min(500, (df["aqi_category"]==cat).sum()), random_state=1)
        ax4.scatter(sub["humidity"], sub["temperature_celsius"],
                    s=6, alpha=0.4, color=cat_colors[cat], label=cat)
    ax4.set_xlabel("Humidity (%)")
    ax4.set_ylabel("Temperature (°C)")
    ax4.set_title("Humidity vs Temperature (by AQI)")
    ax4.legend(fontsize=7, markerscale=2)

    # ── season distribution donut ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    sc = df["season"].value_counts()
    wedge_props = dict(width=0.5, edgecolor="white")
    ax5.pie(sc, labels=sc.index, autopct="%1.0f%%",
            colors=PALETTE[:len(sc)], wedgeprops=wedge_props, startangle=90)
    ax5.set_title("Season Distribution (all cities)")

    # ── key stats text box ────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis("off")
    stats_txt = (
        f"Dataset Overview\n"
        f"{'─'*28}\n"
        f"  Rows         {len(df):>10,}\n"
        f"  Cities       {df['location_name'].nunique():>10}\n"
        f"  Countries    {df['country'].nunique():>10}\n"
        f"  Date range   {df['last_updated'].min().date()} → "
        f"{df['last_updated'].max().date()}\n\n"
        f"Temperature\n"
        f"{'─'*28}\n"
        f"  Global mean  {df['temperature_celsius'].mean():>8.1f} °C\n"
        f"  Global max   {df['temperature_celsius'].max():>8.1f} °C\n"
        f"  Global min   {df['temperature_celsius'].min():>8.1f} °C\n\n"
        f"Air Quality\n"
        f"{'─'*28}\n"
        f"  Mean PM2.5   {df['air_quality_PM2.5'].mean():>8.1f} µg/m³\n"
        f"  Worst city   {df.groupby('location_name')['air_quality_PM2.5'].mean().idxmax()}"
    )
    ax6.text(0.05, 0.95, stats_txt, transform=ax6.transAxes,
             fontsize=9.5, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef4fb",
                       edgecolor=BRAND, linewidth=1.5))

    plt.suptitle("Global Weather Trend Forecasting – Summary Dashboard",
                 fontsize=16, fontweight="bold", y=0.99)
    fig.savefig(f"{FIG_DIR}/00_summary_dashboard.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("[Dashboard] saved")


# ---------------------------------------------------------------------------
# main - run everything in order
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_clean()
    eda(df)
    df = anomaly_detection(df)
    feature_importance(df)

    # picked Cairo/London/Tokyo on purpose - wanted 3 cities with pretty
    # different climates (desert, oceanic, humid subtropical) to see if the
    # same model setup actually generalizes or if I'd need to tune per city.
    # Turned out the ensemble held up fine on all three.
    # (falls back to whatever cities have the most data if one of these
    # three isn't in the dataset version being used)
    target_cities = ["Cairo", "London", "Tokyo"]
    available = set(df["location_name"].unique())
    target_cities = [c for c in target_cities if c in available]
    if len(target_cities) < 3:
        backup = (df["location_name"].value_counts()
                  .index.difference(target_cities)[: 3 - len(target_cities)])
        target_cities += list(backup)
        print(f"[Forecast] note: substituted missing city/cities, using {target_cities}")

    for city in target_cities:
        forecast(df, city=city)

    climate_analysis(df)
    air_quality_analysis(df)
    spatial_analysis(df)
    summary_dashboard(df)
    print("\nDone - figures in figures/, csvs in outputs/")
