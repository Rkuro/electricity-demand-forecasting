import pandas as pd
import os
import numpy as np
import xgboost as xgb # May need to install libomp - MACOS ==> 'brew install libomp'
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ..data.eia import download_balance
import matplotlib.pyplot as plt
import time

start = time.time()

# Download data
print("Downloading data")
csv_filepaths = download_balance()

dfs = []

print("Loading data")
for filepath in csv_filepaths:
    print(f"Loading {filepath}")
    dfs.append(pd.read_csv(filepath))

# Concat data
df = pd.concat(dfs)

print(f"Raw DF shape: {df.shape}")

# Filter for ISNE (New England)
df = df[df["Balancing Authority"] == "ISNE"]

# Convert timestamp column and sort by time - This is important!
df["timestamp"] = pd.to_datetime(df["UTC Time at End of Hour"])
df.set_index("timestamp", inplace=True)
df = df.sort_index()

# For later - checking their predicted values
ISO_Demand_Forecast_df = df[["Demand Forecast (MW)"]]

# Ensure no NaN values
print(f"Number of nan values for target column: {df['Demand (MW)'].isna().sum()}")
df["Demand (MW)"] = df["Demand (MW)"].interpolate(method="linear")
df["Demand Forecast (MW)"] = df["Demand Forecast (MW)"].interpolate(method="linear")

# Features
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df['quarter'] = df.index.quarter
df['year'] = df.index.year
df["month"] = df.index.month
df['dayofyear'] = df.index.dayofyear

# Create Lag Features (Past Demand)
for lag in [1, 6, 24]:  # Use past 1, 6, and 24-hour demand as features
    df[f"demand_lag_{lag}"] = df["Demand (MW)"].shift(lag)

# Rolling Window Features
df["demand_rolling_mean_6"] = df["Demand (MW)"].rolling(window=6).mean()
df["demand_rolling_mean_24"] = df["Demand (MW)"].rolling(window=24).mean()

print(f"Pre features df {df.shape}")
# Drop NaNs from lagging
df = df.dropna(subset=[f"demand_lag_{lag}" for lag in [1, 6, 24]])
print(f"Features df: {df.shape}")

# Drop unecessary cols
df = df.drop(columns=["Data Date", "Local Time at End of Hour", "UTC Time at End of Hour"])

# Fix types for categorical cols
categorical_columns = ["Balancing Authority", "Region"]  # Add any categorical column
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Checking correlation
correlation = df[["Demand (MW)", "demand_lag_1", "Demand Forecast (MW)"]].corr()
print(correlation)

# Shift net generation columns back 1 hour to ensure they don't match realtime dispatch values
generation_cols = [
    "Net Generation (MW) from Coal",
    "Net Generation (MW) from Natural Gas",
    "Net Generation (MW) from Nuclear",
    "Net Generation (MW) from Wind",
    "Net Generation (MW) from Solar",
    "Total Interchange (MW)"
]
new_gen_cols = []
for col in generation_cols:
    df[f"{col}_lag_1"] = df[col].shift(1)
    new_gen_cols.append(f"{col}_lag_1")

df = df.drop(columns=generation_cols)


# Feature cutoff!
features_to_keep = [
    # Time-based features
    "hour",
    "dayofweek",
    "quarter",
    "year",
    "month",
    "dayofyear",

    # Lag & rolling window features
    # "demand_lag_1",
    "demand_lag_6",
    "demand_lag_24",
    "demand_rolling_mean_6",
    "demand_rolling_mean_24",
]# + new_gen_cols

df = df[features_to_keep + ["Demand (MW)"]]  # Ensure target is kept separately


# ******** Typical Train Test Split ********
# Train-Test Split (80% train, 20% test) - MUST BE SORTED BY TIMESTAMP FIRST (done earlier)
cutoff_date = df.index[int(len(df) * 0.8)]
train_df = df[df.index < cutoff_date]
test_df = df[df.index >= cutoff_date]
# ******** Typical Train Test Split ********

# # ******** STORM ANOMALY CASE ********
# # Testing storm anomaly case - October 29–30, 2017  The combination of Tropical Storm Philippe and an extratropical system resulted in approximately 1.2 million power outages in New England
# storm_start = "2017-10-27"
# storm_end = "2017-11-02"
# # Train only on data **before** the storm
# train_df = df[df.index < storm_start]
# # Test only on data **during** the storm
# test_df = df[(df.index >= storm_start) & (df.index <= storm_end)]
# # ******** STORM ANOMALY CASE ********

print(f"Train df {train_df.shape}. Test df: {test_df.shape}")
print(f"Last train timestamp: {train_df.index[-1]}")
print(f"First test timestamp: {test_df.index[0]}")

# Separate features (X) and target variable (y)
X_train = train_df.drop(columns=["Demand (MW)"])
y_train = train_df["Demand (MW)"]

X_test = test_df.drop(columns=["Demand (MW)"])
y_test = test_df["Demand (MW)"]

# Convert to DMatrix format (optimized for XGBoost)
print("Converting to DMatrix format for XGBoost")
print(f"Train set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")
print("X_train columns:", X_train.columns)
dtrain = xgb.DMatrix(
    X_train,
    label=y_train,
    enable_categorical=True
)
dtest = xgb.DMatrix(
    X_test,
    label=y_test,
    enable_categorical=True
)

# XGBoost Model Parameters
params = {
    "objective": "reg:squarederror",  # Regression task
    "eval_metric": "rmse",  # Root Mean Squared Error
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


# Train XGBoost Model
print("Training model...")
model = xgb.train(params, dtrain, num_boost_round=100)

importance = model.get_score(importance_type='weight')
print(sorted(importance.items(), key=lambda x: x[1], reverse=True))

# Forecasting on Test Set
print("Generating predictions")
forecast = model.predict(dtest)
print(f"Forecast shape: {forecast.shape}")

print("Plotting results")
# Plot results
plt.figure(figsize=(12, 5))

# Plot actual demand
plt.plot(test_df.index, y_test, label="Actual Demand")

# test_df = df[(df.index >= storm_start) & (df.index <= storm_end)]
ISO_Demand_Forecast_df = ISO_Demand_Forecast_df[(ISO_Demand_Forecast_df.index >= storm_start) & (ISO_Demand_Forecast_df.index <= storm_end)]
print(ISO_Demand_Forecast_df.head())
plt.plot(test_df.index, ISO_Demand_Forecast_df["Demand Forecast (MW)"], label="ISO Demand Forecast")

# Plot forecast
plt.plot(test_df.index, forecast, label="Predicted Demand (XGBoost)")

plt.xlabel("Time")
plt.ylabel("Demand (MW)")
plt.title("XGBoost Forecast vs. Actual Demand")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Plot results for the last 72 hours
plt.figure(figsize=(12, 5))

# Plot actual demand (blue)
plt.plot(test_df.index[-72:], y_test.iloc[-72:], label="Actual Demand")

# Plot forecast (red dashed line)
plt.plot(test_df.index[-72:], forecast[-72:], label="Predicted Demand (XGBoost)", linestyle="dashed")

# Plot ISO forecast (from data)
plt.plot(test_df.index[-72:], ISO_Demand_Forecast_df["Demand Forecast (MW)"].iloc[-72:], label="ISO Demand Forecast")

plt.xlabel("Time")
plt.ylabel("Demand (MW)")
plt.title("XGBoost Forecast vs. Actual Demand (Last 72 Hours)")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Compute Residuals
residuals = y_test - forecast

plt.figure(figsize=(12, 5))
plt.plot(residuals, label="Residuals", color="green")
plt.title("Residuals of XGBoost Model")
plt.axhline(y=0, color="black", linestyle="--")
plt.legend()
plt.show()

# Compute Error Metrics
mae = mean_absolute_error(y_test, forecast)
rmse = np.sqrt(mean_squared_error(y_test, forecast))
mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100

# Print results
print(f"📊 Model Evaluation Metrics:")
print(f"🔹 Avg. error in MW:  {mae:.2f} MW")
print(f"🔹 Root mean squared error: {rmse:.2f} MW")
print(f"🔹 Mean absolute percentage error: {mape:.2f}%")


# Resize iso forecast to y test
print(f"ISO Forecast shape: {ISO_Demand_Forecast_df.shape}, y_test shape: {y_test.shape}")
ISO_Demand_Forecast_df = ISO_Demand_Forecast_df.iloc[-len(y_test):]

# Compute error metrics of the ISO forecast model
mae = mean_absolute_error(y_test, ISO_Demand_Forecast_df["Demand Forecast (MW)"])
rmse = np.sqrt(mean_squared_error(y_test, ISO_Demand_Forecast_df["Demand Forecast (MW)"]))
mape = np.mean(np.abs((y_test - ISO_Demand_Forecast_df["Demand Forecast (MW)"]) / y_test)) * 100

# Print ISO
print(f"📊 ISO Model Evaluation Metrics:")
print(f"🔹 Avg. error in MW:  {mae:.2f} MW")
print(f"🔹 Root mean squared error: {rmse:.2f} MW")
print(f"🔹 Mean absolute percentage error: {mape:.2f}%")


# Naiive test
naive_forecast = y_test.shift(1).fillna(method="bfill")
naive_mae = mean_absolute_error(y_test, naive_forecast)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_forecast))
naive_mape = np.mean(np.abs((y_test - naive_forecast) / y_test)) * 100

print(f"📊 Naïve Baseline Metrics:")
print(f"🔹 Avg. error in MW: {naive_mae:.2f} MW")
print(f"🔹 Root mean squared error: {naive_rmse:.2f} MW")
print(f"🔹 Mean absolute percentage error: {naive_mape:.2f}%")

end = time.time()
print(f"Finished in {end-start} seconds")