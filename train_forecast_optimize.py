# train_forecast_optimize.py
# TensorFlow time series forecasting + simple battery cost optimization
# Requires: tensorflow>=2.12, pandas, numpy, scikit-learn

import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

CSV_PATH = "energy_timeseries.csv"   # put the downloaded CSV next to this script
TARGET_COL = "net_load_kwh"          # what we forecast
FORECAST_HORIZON = 24                # predict next 24 hours
LOOKBACK = 168                       # use previous 7 days (168 hours)
BATCH_SIZE = 128
EPOCHS = 15
SEED = 42

# ----------------------------
# 1) Load data
# ----------------------------
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Optional: filter any negative or weird values
for c in ["load_kwh","solar_kwh","net_load_kwh"]:
    if c in df:
        df[c] = df[c].clip(lower=0 if c!="net_load_kwh" else None)

# ----------------------------
# 2) Features & scaling
# ----------------------------
feature_cols = ["net_load_kwh","load_kwh","solar_kwh","temp_c","price_usd_per_kwh","hour","dow","month"]
data = df[feature_cols].astype(float).values

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# We’ll forecast the scaled TARGET_COL, then invert scale later.
target_idx = feature_cols.index(TARGET_COL)

# ----------------------------
# 3) Windowing: create sequences X (lookback) -> y (next horizon of target)
# ----------------------------
def make_windows(arr, lookback, horizon, target_col_idx):
    X, y = [], []
    # We align each sample so that X covers [i-lookback, i) and y covers [i, i+horizon)
    for i in range(lookback, len(arr)-horizon):
        X.append(arr[i-lookback:i, :])
        y.append(arr[i:i+horizon, target_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = make_windows(data_scaled, LOOKBACK, FORECAST_HORIZON, target_idx)

# Train/val/test split by time (no shuffling)
n = len(X)
n_train = int(n*0.7)
n_val = int(n*0.15)
X_train, y_train = X[:n_train], y[:n_train]
X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

# ----------------------------
# 4) Build a compact model (Conv1D + LSTM + Dense)
# ----------------------------
tf.keras.utils.set_random_seed(SEED)
inp = tf.keras.Input(shape=(LOOKBACK, X.shape[-1]))
x = tf.keras.layers.Conv1D(32, 5, padding="causal", activation="relu")(inp)
x = tf.keras.layers.Conv1D(32, 5, padding="causal", activation="relu")(x)
x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
out = tf.keras.layers.Dense(FORECAST_HORIZON)(x)

model = tf.keras.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.summary()

# ----------------------------
# 5) Train
# ----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
]
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks
)

# ----------------------------
# 6) Evaluate & forecast the last window (next 24h)
# ----------------------------
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_loss:.4f} | Test MAE: {test_mae:.4f} (scaled units)")

# Forecast next-24h using the last available lookback window
last_window = data_scaled[-LOOKBACK:, :][None, ...]  # shape (1, LOOKBACK, features)
pred_scaled = model.predict(last_window, verbose=0)[0]  # shape (24,)

# Invert scale for target only
# Build helper to invert only target dimension
def invert_target_scaling(pred_scaled_1d):
    # Create zero row, put predictions into target column one by one to inverse transform
    inv = []
    mean = scaler.mean_[target_idx]
    scale = scaler.scale_[target_idx]
    return pred_scaled_1d * scale + mean

y_pred_next24 = invert_target_scaling(pred_scaled)

# Prepare a nice forecast DataFrame for the next 24 hours
last_ts = df["timestamp"].iloc[-1]
future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=FORECAST_HORIZON, freq="H")
forecast_df = pd.DataFrame({"timestamp": future_index, "forecast_net_load_kwh": y_pred_next24})

print("\nNext-24h forecast (first few rows):")
print(forecast_df.head())

# ----------------------------
# 7) Simple battery optimization using forecast + prices
# ----------------------------
# We choose a single-home battery model:
# - Capacity: 13.5 kWh
# - Max charge/discharge power: 5 kW
# - Round-trip efficiency: 90% (charge_eff=0.95, discharge_eff=0.95)
# - Objective: minimize cost of (net_load + battery_flow) * price over the next 24h
#   Positive battery_flow means *discharging* (supplying load), negative means charging.

BATTERY_CAP_KWH = 13.5
P_MAX_KW = 5.0
CHARGE_EFF = 0.95
DISCHARGE_EFF = 0.95
INITIAL_SOC = 0.5 * BATTERY_CAP_KWH
FINAL_SOC_TARGET = 0.5 * BATTERY_CAP_KWH  # keep neutral SoC by end (optional)

# Get corresponding next-24h prices (we’ll extend dataframe with a merge on timestamp)
# First, align prices from original df to the future timestamps by rolling forward last known pattern.
price_series = df.set_index("timestamp")["price_usd_per_kwh"]
# naive approach: repeat last 24h price pattern
last_24_prices = price_series.iloc[-24:].values
future_prices = np.resize(last_24_prices, FORECAST_HORIZON)
forecast_df["price_usd_per_kwh"] = future_prices

# Greedy heuristic:
# Sort hours by price ascending for charging and descending for discharging
# and iterate until marginal value equalizes within constraints.
# Simpler and fast: two-pass heuristic:
soc = INITIAL_SOC
battery_flow = np.zeros(FORECAST_HORIZON)  # + discharge to meet load, - charge

# Strategy:
# 1) Identify low-price hours -> charge (respecting power, capacity, charge eff).
# 2) Identify high-price hours -> discharge (respecting power, capacity, discharge eff).
# Use price thresholds as quantiles.
low_thr = np.quantile(future_prices, 0.35)
high_thr = np.quantile(future_prices, 0.65)

# First pass: charge on low hours
for t in range(FORECAST_HORIZON):
    price_t = future_prices[t]
    if price_t <= low_thr:
        max_charge = min(P_MAX_KW, (BATTERY_CAP_KWH - soc) / CHARGE_EFF)
        # Don't overcharge, and avoid charging if forecast net load is negative (export) to keep simple
        if max_charge > 0:
            battery_flow[t] = -max_charge  # negative = charging
            soc += (-battery_flow[t]) * CHARGE_EFF

# Second pass: discharge on high hours
for t in range(FORECAST_HORIZON):
    price_t = future_prices[t]
    if price_t >= high_thr:
        max_discharge = min(P_MAX_KW, soc * DISCHARGE_EFF)  # kWh available this hour
        if max_discharge > 0:
            battery_flow[t] = max(battery_flow[t], max_discharge)  # positive = discharging
            soc -= battery_flow[t] / DISCHARGE_EFF

# Optional: gently steer SoC toward target in the remaining medium-price hours
for t in range(FORECAST_HORIZON):
    price_t = future_prices[t]
    if low_thr < price_t < high_thr:
        # small nudge to move SOC towards target
        diff = FINAL_SOC_TARGET - soc
        if abs(diff) > 0.2:
            if diff > 0:
                # need to charge a bit
                amt = min(P_MAX_KW/2, (BATTERY_CAP_KWH - soc) / CHARGE_EFF)
                battery_flow[t] = min(battery_flow[t], -amt)
                soc += (-min(0, battery_flow[t])) * CHARGE_EFF
            else:
                # need to discharge a bit
                amt = min(P_MAX_KW/2, soc * DISCHARGE_EFF)
                battery_flow[t] = max(battery_flow[t], amt)
                soc -= max(0, battery_flow[t]) / DISCHARGE_EFF

optimized_df = forecast_df.copy()
optimized_df["battery_flow_kwh"] = battery_flow  # +discharge, -charge
optimized_df["optimized_net_grid_kwh"] = optimized_df["forecast_net_load_kwh"] - optimized_df["battery_flow_kwh"]
optimized_df["cost_usd_no_battery"] = optimized_df["forecast_net_load_kwh"] * optimized_df["price_usd_per_kwh"]
optimized_df["cost_usd_with_battery"] = optimized_df["optimized_net_grid_kwh"] * optimized_df["price_usd_per_kwh"]

savings = optimized_df["cost_usd_no_battery"].sum() - optimized_df["cost_usd_with_battery"].sum()
print(f"\nEstimated 24h cost savings with battery strategy: ${savings:.2f}")

# Save artifacts
forecast_path = "next24_forecast.csv"
opt_path = "next24_optimized_schedule.csv"
forecast_df.to_csv(forecast_path, index=False)
optimized_df.to_csv(opt_path, index=False)
print(f"\nSaved:\n - {forecast_path}\n - {opt_path}")

# Tip: to visualize, you can plot 'forecast_net_load_kwh' vs 'optimized_net_grid_kwh' and prices.
# ----------------------------
# 8) Visualization with Matplotlib (User-Friendly)
# ----------------------------
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")  # clean style

# --- Forecast vs Optimized Net Load ---
plt.figure(figsize=(12, 6))
plt.plot(optimized_df["timestamp"], optimized_df["forecast_net_load_kwh"],
         label="Forecast Net Load (kWh)", color="blue", marker="o")
plt.plot(optimized_df["timestamp"], optimized_df["optimized_net_grid_kwh"],
         label="Optimized Net Grid Load (kWh)", color="green", linestyle="--", marker="s")
plt.xticks(rotation=45)
plt.ylabel("kWh")
plt.title("Next-24h Energy Demand Forecast vs Battery-Optimized Net Grid Load")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Battery Flow vs Electricity Price (Combo Chart) ---
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(optimized_df["timestamp"], optimized_df["battery_flow_kwh"], color="orange", alpha=0.6, label="Battery Flow (kWh)")
ax1.set_ylabel("Battery Flow (kWh)", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

ax2 = ax1.twinx()
ax2.plot(optimized_df["timestamp"], optimized_df["price_usd_per_kwh"], color="blue", marker="o", label="Price (USD/kWh)")
ax2.set_ylabel("Price (USD/kWh)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")

fig.autofmt_xdate()
plt.title("Battery Charging/Discharging vs Electricity Price")
fig.tight_layout()
plt.show()

# --- Cumulative Cost Comparison ---
plt.figure(figsize=(12, 6))
plt.plot(optimized_df["timestamp"], optimized_df["cost_usd_no_battery"].cumsum(),
         label="Cumulative Cost (No Battery)", color="red", linestyle="--", marker="o")
plt.plot(optimized_df["timestamp"], optimized_df["cost_usd_with_battery"].cumsum(),
         label="Cumulative Cost (With Battery)", color="green", marker="s")
plt.xticks(rotation=45)
plt.ylabel("USD")
plt.title("24h Energy Cost Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- KPI Cards as Text ---
total_no_batt = optimized_df["cost_usd_no_battery"].sum()
total_with_batt = optimized_df["cost_usd_with_battery"].sum()
total_savings = total_no_batt - total_with_batt

print("\n=== 24h Energy Cost Summary ===")
print(f"Total Cost Without Battery: ${total_no_batt:.2f}")
print(f"Total Cost With Battery:    ${total_with_batt:.2f}")
print(f"Total Savings Achieved:     ${total_savings:.2f}")

# ----------------------------
# 9) Model Accuracy Check
# ----------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predict on test set
y_pred_test_scaled = model.predict(X_test, verbose=0)  # shape (num_samples, FORECAST_HORIZON)
# Flatten both arrays for metrics (hour-level comparison)
y_pred_test_flat = invert_target_scaling(y_pred_test_scaled.flatten())
y_true_test_flat = invert_target_scaling(y_test.flatten())

# Compute metrics
mae = mean_absolute_error(y_true_test_flat, y_pred_test_flat)
mse = mean_squared_error(y_true_test_flat, y_pred_test_flat)
rmse = mse**0.5
mape = np.mean(np.abs((y_true_test_flat - y_pred_test_flat) / y_true_test_flat)) * 100

print("\n=== Model Accuracy on Test Set ===")
print(f"MAE  : {mae:.2f} kWh")
print(f"RMSE : {rmse:.2f} kWh")
print(f"MAPE : {mape:.2f} %")

# ----------------------------
# 10) Plot Predicted vs Actual Net Load (Test Set)
# ----------------------------
import matplotlib.pyplot as plt

# Plot first 100 hours for clarity
plt.figure(figsize=(12,6))
plt.plot(y_true_test_flat[:100], label="Actual Net Load", marker="o", color="blue")
plt.plot(y_pred_test_flat[:100], label="Predicted Net Load", marker="s", color="orange")
plt.title("Model Accuracy Check (First 100 Hours of Test Set)")
plt.xlabel("Hour Index")
plt.ylabel("Net Load (kWh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
