# Smart Energy Forecasting & Demand Optimization

## Why this project?
With increasing adoption of renewable energy sources like solar and wind, microgrids need smart forecasting to manage energy efficiently. Traditional methods cannot predict variable energy generation accurately, which can lead to wasted energy or power shortages.

## What problem does it solve?
- Predicts short-term solar and wind generation using historical data.
- Provides actionable recommendations for energy storage and load balancing.
- Helps microgrid operators optimize energy usage, reduce costs, and improve sustainability.

## Approach
1. **Data Collection & Preprocessing**
   - Uses microgrid data including solar and wind generation.
   - Handles missing values and scales features for better model performance.

2. **Time Series Forecasting**
   - Builds an LSTM-based model using TensorFlow/Keras.
   - Uses past 24 hours of data to predict the next time step.
     
<img src="images/put readme1.png" alt="Forecast Example 1" width="60%" style="display:block; margin:auto;">
<br>
<img src="images/put readme2.png" alt="Forecast Example 2" width="60%" style="display:block; margin:auto;">

3. **Prediction & Optimization**
   - Forecasts future energy generation.
   - Compares predicted vs actual generation and recommends whether to store energy or use it immediately.
     
   <p align="center">
     <img src="images/put readme3.png" alt="Prediction Example 1" width="45%">
   </p>
4. **Visualization**
   - Plots actual vs predicted solar and wind generation.
   - Interactive options allow selecting which features to visualize.

## Future Work
- Extend to predict total load/demand.
- Deploy as a web dashboard using Streamlit or Dash.
- Add anomaly detection for unexpected generation spikes.
