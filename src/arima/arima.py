import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import time
import os

start = time.time()

base_data_dir = "/Users/robinkurosawa/git/electricity-demand-forecasting/data/raw"

Jan_24_CSV_FILE = os.path.join(base_data_dir, "EIA930_BALANCE_2024_Jan_Jun.csv")
Jul_24_CSV_FILE = os.path.join(base_data_dir, "EIA930_BALANCE_2024_Jul_Dec.csv")
Jan_25_CSV_FILE = os.path.join(base_data_dir, "EIA930_BALANCE_2025_Jan_Jun.csv")

# Load data
df_1 = pd.read_csv(Jan_24_CSV_FILE)
df_2 = pd.read_csv(Jul_24_CSV_FILE)
df_3 = pd.read_csv(Jan_25_CSV_FILE)

# Concat data
df = pd.concat([df_1, df_2, df_3])

# Filter for New England region (ISNE)
df = df[df["Balancing Authority"] == "ISNE"]

# Convert timestamp column
df["timestamp"] = pd.to_datetime(df["UTC Time at End of Hour"])

# Set timestamp as index
df.set_index("timestamp", inplace=True)

# Sort by time
df = df.sort_index()

# Ensure that there are no NaN values - interpolate missing values linearly
# Not sure why there are NaN values to begin with...
df["Demand (MW)"] = df["Demand (MW)"].interpolate(method="linear")

ISO_Demand_Forecast_df = df[["Demand Forecast (MW)"]]

# Select target variable (Actual Demand)
df = df[["Demand (MW)"]]

# Split into train/test (NOT random due to timeseries!)
train_size = int(len(df) * 0.8)  # 80% training data
train_df = df.iloc[:train_size]  # First 80%
test_df = df.iloc[train_size:]   # Last 20% for validation

ISO_Demand_Forecast_df = ISO_Demand_Forecast_df.iloc[train_size:]

# Print stats prior to fit
print("Dataset Shape")
print(df.shape)

# Find a 'D' value
result = adfuller(df["Demand (MW)"])
print("ADF Statistic:", result)

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_acf(df["Demand (MW)"], lags=48, ax=axes[0])  # ACF up to 48 hours
axes[0].set_title("Autocorrelation Function (ACF)")

plot_pacf(df["Demand (MW)"], lags=48, ax=axes[1])  # PACF up to 48 hours
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.show()

print("Training ARIMA Model...")
# Fit ARIMA model
model = ARIMA(
    train_df["Demand (MW)"],
    order=(2, 0, 2), # (p, d, q)
    freq='h',
    seasonal_order=(2, 1, 2, 24)
)
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

print("Inferencing validation set...")
# Forecast the test dataset
forecast = model_fit.forecast(steps=len(test_df))

# Plot results
plt.figure(figsize=(12, 5))

# Plot actual demand (blue)
plt.plot(test_df.index, test_df["Demand (MW)"], label="Actual Demand")

# Plot forecast (red dashed line)
plt.plot(test_df.index, forecast, label="Predicted Demand (ARIMA)")

# Plot ISO forecast (from data)
plt.plot(test_df.index, ISO_Demand_Forecast_df["Demand Forecast (MW)"], label="ISO Demand Forecast")

plt.xlabel("Time")
plt.ylabel("Demand (MW)")
plt.title("ARIMA Forecast vs. Actual Demand")
plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.show()

# Plot results for the last 72 hours
plt.figure(figsize=(12, 5))

# Plot actual demand (blue)
plt.plot(test_df.index[-72:], test_df["Demand (MW)"].iloc[-72:], label="Actual Demand")

# Plot forecast (red dashed line)
plt.plot(test_df.index[-72:], forecast[-72:], label="Predicted Demand (ARIMA)")

# Plot ISO forecast (from data)
plt.plot(test_df.index[-72:], ISO_Demand_Forecast_df["Demand Forecast (MW)"].iloc[-72:], label="ISO Demand Forecast")

plt.xlabel("Time")
plt.ylabel("Demand (MW)")
plt.title("ARIMA Forecast vs. Actual Demand (Last 72 Hours)")
plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.show()

residuals = model_fit.resid
plt.figure(figsize=(12,5))
plt.plot(residuals)
plt.title("Residuals of ARIMA Model")
plt.show()

# Compute error metrics
mae = mean_absolute_error(test_df["Demand (MW)"], forecast)
rmse = np.sqrt(mean_squared_error(test_df["Demand (MW)"], forecast))
mape = np.mean(np.abs((test_df["Demand (MW)"] - forecast) / test_df["Demand (MW)"])) * 100

# Print results
print(f"📊 Model Evaluation Metrics:")
print(f"🔹 Avg. error in MW:  {mae:.2f} MW")
print(f"🔹 Root mean squared error: {rmse:.2f} MW")
print(f"🔹 Mean absolute percentage error: {mape:.2f}%")

print("===================================================================================")

# Compute error metrics of the ISO forecast model
mae = mean_absolute_error(test_df["Demand (MW)"], ISO_Demand_Forecast_df["Demand Forecast (MW)"])
rmse = np.sqrt(mean_squared_error(test_df["Demand (MW)"], ISO_Demand_Forecast_df["Demand Forecast (MW)"]))
mape = np.mean(np.abs((test_df["Demand (MW)"] - ISO_Demand_Forecast_df["Demand Forecast (MW)"]) / test_df["Demand (MW)"])) * 100

# Print
print(f"📊 ISO Model Evaluation Metrics:")
print(f"🔹 Avg. error in MW:  {mae:.2f} MW")
print(f"🔹 Root mean squared error: {rmse:.2f} MW")
print(f"🔹 Mean absolute percentage error: {mape:.2f}%")

end = time.time()
print(f"Finished in {end - start} seconds")