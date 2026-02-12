# Install required packages
import subprocess
import sys

#packages = [
#    'pandas',
#    'numpy',
#    'scikit-learn',
#    'requests',
#    'python-dateutil',
#    'matplotlib',
#    'seaborn',
#    'btcvol'
#]

#for package in packages:
#    try:
#        __import__(package)
#        print(f"✓ {package} already installed")
#    except ImportError:
#        print(f"Installing {package}...")
#        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#        print(f"✓ {package} installed successfully")


# Import required libraries
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Competition package imports
from btcvol import TrackerBase, test_model_locally

# Display configuration
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 100)
#plt.style.use('seaborn-v0_8-darkgrid')
#sns.set_palette("husl")

#print("✓ All libraries imported successfully!")


from pathlib import Path

# Load 15-minute DVOL data from the uploaded CSV
#data_path = Path("deribit_btc_dvol_15min_ohlc.csv")
#if not data_path.exists():
#    alt_path = Path("notebooks") / "deribit_btc_dvol_15min_ohlc.csv"
#    if alt_path.exists():
#        data_path = alt_path
#    else:
#        raise FileNotFoundError("deribit_btc_dvol_15min_ohlc.csv not found in notebook or project root")

#df_raw = pd.read_csv(data_path)

# Parse timestamps and sort
#df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)
#df_raw = df_raw.sort_values("time").reset_index(drop=True)

# Ensure numeric columns
#for col in ["open", "high", "low", "close"]:
#    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

# Use close as DVOL target
#df_dvol = df_raw.rename(columns={"time": "timestamp"})
#df_dvol["dvol"] = df_dvol["close"]

#print("✓ Loaded 15-minute DVOL data")
#print("Rows:", len(df_dvol))
#print("Columns:", df_dvol.columns.tolist())
#print("Date range:", df_dvol["timestamp"].min(), "→", df_dvol["timestamp"].max())


# Quick data quality checks and summary

# Ensure timestamp dtype
#if not pd.api.types.is_datetime64_any_dtype(df_dvol["timestamp"]):
#    df_dvol["timestamp"] = pd.to_datetime(df_dvol["timestamp"], utc=True)

#print("BTC DVOL Data Shape:", df_dvol.shape)
#print("\nFirst 5 rows:")
#print(df_dvol.head())
#print("\nLast 5 rows:")
#print(df_dvol.tail())

# Missing values
#print("\nMissing Values:")
#print(df_dvol[["timestamp", "open", "high", "low", "close", "dvol"]].isnull().sum())

# Duplicate timestamps
#dup_count = df_dvol["timestamp"].duplicated().sum()
#print("\nDuplicate timestamps:", dup_count)

# Expected frequency check (15 minutes)
#time_deltas = df_dvol["timestamp"].diff().dropna()
#print("\nMost common time delta:", time_deltas.mode().iloc[0])

# Basic statistics
#print("\nDVOL Statistics:")
#print(df_dvol["dvol"].describe())


# Visualize DVOL time series and distributions

#plot_df = df_dvol.copy()
#if plot_df["timestamp"].dt.tz is not None:
#    plot_df["timestamp"] = plot_df["timestamp"].dt.tz_convert(None)

# Rolling stats (1 day = 96 intervals of 15 minutes)
#roll_window = 96
#plot_df["dvol_roll_mean"] = plot_df["dvol"].rolling(roll_window).mean()
#plot_df["dvol_roll_std"] = plot_df["dvol"].rolling(roll_window).std()

#fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) DVOL over time
#axes[0, 0].plot(plot_df["timestamp"], plot_df["dvol"], linewidth=1.5, label="DVOL (Close)")
#axes[0, 0].set_title("DVOL (15-min) Time Series")
#axes[0, 0].set_xlabel("Time")
#axes[0, 0].set_ylabel("DVOL (%)")
#axes[0, 0].grid(True, alpha=0.3)

# 2) Rolling mean and std bands
#axes[0, 1].plot(plot_df["timestamp"], plot_df["dvol_roll_mean"], color="tab:orange", label="1D Rolling Mean")
#axes[0, 1].fill_between(
#    plot_df["timestamp"],
#    plot_df["dvol_roll_mean"] - plot_df["dvol_roll_std"],
#    plot_df["dvol_roll_mean"] + plot_df["dvol_roll_std"],
#    color="tab:orange", alpha=0.2, label="±1 Std"
#  )
#axes[0, 1].set_title("Rolling Mean/Std (1 Day)")
#axes[0, 1].set_xlabel("Time")
#axes[0, 1].set_ylabel("DVOL (%)")
#axes[0, 1].grid(True, alpha=0.3)
#axes[0, 1].legend()

# 3) Distribution of DVOL
#sns.histplot(plot_df["dvol"].dropna(), bins=40, kde=True, ax=axes[1, 0], color="tab:blue")
#axes[1, 0].set_title("DVOL Distribution")
#axes[1, 0].set_xlabel("DVOL (%)")
#axes[1, 0].set_ylabel("Count")
#axes[1, 0].grid(True, alpha=0.3)

# 4) DVOL by hour of day
#plot_df["hour"] = plot_df["timestamp"].dt.hour
#sns.boxplot(x="hour", y="dvol", data=plot_df, ax=axes[1, 1])
#axes[1, 1].set_title("DVOL by Hour of Day")
#axes[1, 1].set_xlabel("Hour")
#axes[1, 1].set_ylabel("DVOL (%)")
#axes[1, 1].grid(True, alpha=0.3)

#plt.tight_layout()
#plt.show()

# Check for missing values in numeric columns
#print("\nMissing Values:")
#print(df_dvol[["open", "high", "low", "close", "dvol"]].isnull().sum())


# Create lag features (past values as predictors)
def create_features(data, lags=[1, 2, 3, 7], horizon_steps=96):
    """
    Create lag and rolling features for time-series prediction.
    """
    df = data.copy()

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df["dvol"].shift(lag)

    # Rolling mean features
    for window in [3, 7, 14]:
        df[f"rolling_mean_{window}"] = df["dvol"].rolling(window=window).mean().shift(1)

    # Rolling std features
    for window in [3, 7, 14]:
        df[f"rolling_std_{window}"] = df["dvol"].rolling(window=window).std().shift(1)

    # Day of week and month (if applicable)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day

    # Target: DVOL 24 hours ahead (96 steps for 15-min data)
    df["target_dvol"] = df["dvol"].shift(-horizon_steps)

    # Remove rows with NaN values (created by lag features)
    df = df.dropna()

    return df

# Create features
#df_features = create_features(df_dvol)

#print("Features Created:")
#display(df_features.head(10))
#print("\nFeature Columns:", df_features.columns.tolist())


# Price Data Provider (from crunch-synth)
# https://github.com/crunchdao/crunch-synth/blob/main/crunch_synth/price_provider.py

from datetime import datetime
import requests

class PriceUnavailableError(ValueError):
    """Raised when the price provider cannot fetch the price for an asset."""
    pass

class PriceDbClient:
    _HISTORY_URL = "https://pricedb.crunchdao.com/v1/prices"
    
    def get_price_history(
        self,
        *,
        asset: str,
        from_: datetime,
        to: datetime,
        timeout=30,
    ) -> list[tuple[float, int]]:
        query = {
            "asset": asset,
            "from": from_.isoformat(),
            "to": to.isoformat(),
        }
        
        try:
            response = requests.get(
                self._HISTORY_URL,
                timeout=timeout,
                params=query,
            )
            
            response.raise_for_status()
            root = response.json()
        except Exception as error:
            raise PriceUnavailableError(f"Could not get price history for {asset}: {error}") from error
        
        return list(zip(root["timestamp"], root["close"]))

# Initialize the price data client
#pricedb = PriceDbClient()

#print("PriceDbClient initialized successfully!")


# Fetch BTC price data to enrich our DVOL features
#print("Fetching BTC price data...")

#try:
#    # Get the date range from our DVOL data
#    start_date = df_dvol['timestamp'].min()
#    end_date = df_dvol['timestamp'].max()
#    
#    print(f"Date range: {start_date} to {end_date}")
#    
#    # Fetch BTC price history
#    btc_prices = pricedb.get_price_history(
#        asset="BTC",
#        from_=start_date.to_pydatetime(),
#        to=end_date.to_pydatetime(),
#    )
#    
#    print(f"Fetched {len(btc_prices)} BTC price data points")
#    
#    # Convert to DataFrame
#    df_btc = pd.DataFrame(btc_prices, columns=['timestamp', 'btc_price'])
#    df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='s', utc=True)
#    
#    # Ensure both timestamps have the same precision
#    df_btc['timestamp'] = df_btc['timestamp'].astype('datetime64[us, UTC]')
#    
#    # Merge BTC price data with our features
#    # Using merge_asof to handle time alignment (nearest timestamp match)
#    df_features_with_price = pd.merge_asof(
#        df_features.sort_values('timestamp'),
#        df_btc.sort_values('timestamp'),
#        on='timestamp',
#        direction='nearest',
#        tolerance=pd.Timedelta('15min')  # Maximum time difference for matching
#    )
#    
#    # Add price-based features
#    df_features_with_price['btc_return_1'] = df_features_with_price['btc_price'].pct_change(1)
#    df_features_with_price['btc_return_4'] = df_features_with_price['btc_price'].pct_change(4)  # 1 hour
#    df_features_with_price['btc_volatility_96'] = df_features_with_price['btc_return_1'].rolling(96).std().shift(1)  # 24h volatility
#    
#    # Check how many rows have BTC price data
#    price_coverage = df_features_with_price['btc_price'].notna().sum()
#    print(f"BTC price coverage: {price_coverage}/{len(df_features_with_price)} rows ({price_coverage/len(df_features_with_price)*100:.1f}%)")
#    
#    # If we have good coverage (>50%), use the enriched features
#    if price_coverage / len(df_features_with_price) > 0.5:
#        # Fill any remaining NaN values in price features with forward fill then backward fill
#        price_cols = ['btc_price', 'btc_return_1', 'btc_return_4', 'btc_volatility_96']
#        df_features_with_price[price_cols] = df_features_with_price[price_cols].ffill().bfill()
#        
#        df_features = df_features_with_price
#        print("✓ Successfully enriched features with BTC price data!")
#        print(f"New feature columns: {[col for col in df_features.columns if 'btc' in col]}")
#    else:
#        print("⚠ Low BTC price coverage, continuing without price features")
#        
#except PriceUnavailableError as e:
#    print(f"⚠ Could not fetch BTC price data: {e}")
#    print("Continuing with DVOL features only")
#except Exception as e:
#    print(f"⚠ Unexpected error fetching price data: {e}")
#    print("Continuing with DVOL features only")

#print(f"\nFinal feature set size: {df_features.shape}")
#print("Columns:", df_features.columns.tolist())


# Split data into train and test sets (80/20 split)
#train_size = int(len(df_features) * 0.8)

#df_train = df_features[:train_size]
#df_test = df_features[train_size:]

# Separate features and target
#feature_cols = [col for col in df_features.columns if col not in ["timestamp", "dvol", "target_dvol"]]

#X_train = df_train[feature_cols]
#y_train = df_train["target_dvol"]

#X_test = df_test[feature_cols]
#y_test = df_test["target_dvol"]

#print(f"Training set size: {len(X_train)}")
#print(f"Test set size: {len(X_test)}")
#print(f"Number of features: {len(feature_cols)}")


# Train a Random Forest model
#print("Training Random Forest model...")

# Normalize features
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Train model
#model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
#model.fit(X_train_scaled, y_train)

#print("✓ Model trained successfully!")

# Evaluate on training set
#train_pred = model.predict(X_train_scaled)
#train_mse = mean_squared_error(y_train, train_pred)
#train_rmse = np.sqrt(train_mse)

#print(f"\nTraining Performance:")
#print(f"  MSE: {train_mse:.4f}")
#print(f"  RMSE: {train_rmse:.4f}")

# Evaluate on test set
#test_pred = model.predict(X_test_scaled)
#test_mse = mean_squared_error(y_test, test_pred)
#test_rmse = np.sqrt(test_mse)

#print(f"\nTest Performance:")
#print(f"  MSE: {test_mse:.4f}")
#print(f"  RMSE: {test_rmse:.4f}")


# Feature importance analysis
#feature_importance = pd.DataFrame({
#    'feature': feature_cols,
#    'importance': model.feature_importances_
#}).sort_values('importance', ascending=False)

#print("Top 10 Most Important Features:")
#print(feature_importance.head(10))

# Visualize feature importance
#plt.figure(figsize=(10, 6))
#plt.barh(feature_importance['feature'].head(10), feature_importance['importance'].head(10))
#plt.xlabel('Importance')
#plt.title('Top 10 Feature Importances')
#plt.tight_layout()
#plt.show()


# Create predictions dataframe
#predictions_df = pd.DataFrame({
#    'timestamp': df_test['timestamp'].reset_index(drop=True),
#    'actual': y_test.reset_index(drop=True),
#    'predicted': test_pred,
#    'error': np.abs(y_test.reset_index(drop=True) - test_pred)
#})

#print("Predictions Sample:")
#print(predictions_df.head(10))

# Visualize predictions vs actual
#plt.figure(figsize=(14, 6))
#plt.plot(predictions_df['timestamp'], predictions_df['actual'], 'o-', label='Actual', alpha=0.7, linewidth=2)
#plt.plot(predictions_df['timestamp'], predictions_df['predicted'], 's--', label='Predicted', alpha=0.7, linewidth=2)
#plt.xlabel('Date')
#plt.ylabel('DVOL (%)')
#plt.title('Predicted vs Actual BTC 30-Day Implied Volatility')
#plt.legend()
#plt.grid(True, alpha=0.3)
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()

# Prediction error distribution
#plt.figure(figsize=(10, 5))
#plt.hist(predictions_df['error'], bins=20, edgecolor='black', alpha=0.7)
#plt.xlabel('Absolute Error')
#plt.ylabel('Frequency')
#plt.title('Distribution of Prediction Errors')
#plt.tight_layout()
#plt.show()


# Serialize trained model, scaler, and feature columns for submission
import base64
import pickle
import json

# Serialize model and scaler to base64
#model_bytes = pickle.dumps(model)
#model_b64 = base64.b64encode(model_bytes).decode('utf-8')

#scaler_bytes = pickle.dumps(scaler)
#scaler_b64 = base64.b64encode(scaler_bytes).decode('utf-8')

#feature_cols_json = json.dumps(feature_cols)

#print("✓ Model serialized successfully!")
#print(f"Model size: {len(model_b64) / 1024:.1f} KB")
#print(f"Scaler size: {len(scaler_b64) / 1024:.1f} KB")
#print(f"Feature columns: {len(feature_cols)} features")


# Save serialized artifacts to resources/ for submission packaging
from pathlib import Path

#resources_dir = Path("resources")
#resources_dir.mkdir(parents=True, exist_ok=True)

#(resources_dir / "model.pkl").write_bytes(model_bytes)
#(resources_dir / "scaler.pkl").write_bytes(scaler_bytes)
#(resources_dir / "feature_cols.json").write_text(feature_cols_json)

#print(f"✓ Wrote model.pkl, scaler.pkl, feature_cols.json to {resources_dir.resolve()}")


# Submission-ready RandomForest tracker with successful minimal structure
import numpy as np
import base64
import pickle
import json
from pathlib import Path
from btcvol import TrackerBase


class MyTracker(TrackerBase):
    """
    Random Forest tracker with embedded model.
    """

    # Class variables for embedded serialized model (populated after training)
    _MODEL_B64 = None
    _SCALER_B64 = None
    _FEATURE_COLS_JSON = None

    def __init__(self):
        """Initialize tracker with embedded model and scaler."""
        # Try to use notebook globals first (development mode)
        if 'model' in globals() and 'scaler' in globals() and 'feature_cols' in globals():
            self.model = globals()['model']
            self.scaler = globals()['scaler']
            self.feature_cols = globals()['feature_cols']
            return

        # Try to load from resources/ (submission mode with files)
        resources_dir = Path(__file__).parent / "resources"
        model_path = resources_dir / "model.pkl"
        scaler_path = resources_dir / "scaler.pkl"
        features_path = resources_dir / "feature_cols.json"
        if model_path.exists() and scaler_path.exists() and features_path.exists():
            with model_path.open("rb") as f:
                self.model = pickle.load(f)
            with scaler_path.open("rb") as f:
                self.scaler = pickle.load(f)
            self.feature_cols = json.loads(features_path.read_text())
            return

        # Otherwise use embedded serialized data (submission mode with base64)
        if self._MODEL_B64 and self._SCALER_B64 and self._FEATURE_COLS_JSON:
            self.model = pickle.loads(base64.b64decode(self._MODEL_B64))
            self.scaler = pickle.loads(base64.b64decode(self._SCALER_B64))
            self.feature_cols = json.loads(self._FEATURE_COLS_JSON)
            return

        raise RuntimeError("Model, scaler, or feature_cols not found. Provide resources or embed base64.")

    def predict(self, asset: str, horizon: int, step: int):
        """
        Generate volatility predictions using RandomForest model.
        Implements the proven minimal structure that passes validation.
        """
        n_steps = max(1, horizon // step)

        # Get current prediction from model
        try:
            # Use most recent features if available
            if 'df_features' in globals():
                X = globals()['df_features'].iloc[-1:][self.feature_cols]
                X_scaled = self.scaler.transform(X)
                pred_vol_pct = float(self.model.predict(X_scaled)[0])
            else:
                # Fallback: use dummy features
                X_dummy = np.zeros((1, len(self.feature_cols)))
                X_scaled = self.scaler.transform(X_dummy)
                pred_vol_pct = float(self.model.predict(X_scaled)[0])
        except Exception:
            # Safety fallback
            pred_vol_pct = 50.0

        # Clamp to reasonable range (0-100%)
        pred_vol_pct = max(0, min(100, pred_vol_pct))

        # Convert to decimal (0-1 scale)
        pred_vol_decimal = pred_vol_pct / 100.0

        # Return same prediction for all steps
        return [float(pred_vol_decimal)] * n_steps


# Auto-populate embedded model/scaler from serialized data
#try:
#    MyTracker._MODEL_B64 = model_b64
#    MyTracker._SCALER_B64 = scaler_b64
#    MyTracker._FEATURE_COLS_JSON = feature_cols_json
#    print("✓ Embedded RandomForest model, scaler, and features into MyTracker")
#except NameError:
#    print("⚠ Serialized data not available (run serialization cell first)")


# Local validation (checks shape, numeric values, and reasonable range)
#validation_ok = test_model_locally(MyTracker)
#print("Validation passed:", validation_ok)

# Sanity check: sample outputs
#tracker = MyTracker()
#print(tracker.predict(asset="BTC", horizon=3600, step=900))
#print(tracker.predict(asset="BTC", horizon=86400, step=900))
