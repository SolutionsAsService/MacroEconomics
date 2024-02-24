import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load historical price data
data = pd.read_csv('historical_prices.csv', parse_dates=['Date'], index_col='Date')

# Visualize the historical price data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Price'], label='Historical Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Historical Price Data')
plt.legend()
plt.show()

# Prepare the data for linear regression
X = np.array(range(len(data))).reshape(-1, 1)  # Use days as the independent variable
y = data['Price'].values  # Use price as the dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future prices
future_days = 30  # Predict prices for the next 30 days
future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1)[1:]
future_X = np.array(range(len(data), len(data) + future_days)).reshape(-1, 1)
future_predictions = model.predict(future_X)

# Visualize the historical prices and future predictions
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Price'], label='Historical Prices')
plt.plot(future_dates, future_predictions, label='Predicted Future Prices', linestyle='--', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Prediction')
plt.legend()
plt.show()
