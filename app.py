from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

# Load or train the model
def train_model():
    # Load the datasets
    stock_prices = pd.read_csv(r"C:\Users\KIIT\OneDrive\Documents\AD Lab\Lab3\historical_stock_prices.csv")
    stocks = pd.read_csv(r"C:\Users\KIIT\OneDrive\Documents\AD Lab\Lab3\historical_stocks.csv")

    # Prepare data for training
    merged_data = pd.merge(stock_prices, stocks, on='ticker')
    stock_data = merged_data[merged_data['ticker'] == 'AAPL']
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values(by='date')
    stock_data['Target'] = stock_data['close'].shift(-1)
    stock_data = stock_data.dropna()

    # Features and target
    features = ['close', 'volume']
    X = stock_data[features]
    y = stock_data['Target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'linear_regression_model.pkl')
    return model

# Check if a model exists, otherwise train it
if os.path.exists('linear_regression_model.pkl'):
    model = joblib.load('linear_regression_model.pkl')
else:
    model = train_model()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data
        data = request.get_json()
        close = float(data['close'])
        volume = float(data['volume'])

        # Make prediction
        prediction = model.predict([[close, volume]])[0]
        return jsonify({'predicted_price': prediction})
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'Prediction failed.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
