#without feature

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import numpy as np
import matplotlib.pyplot as plt

# Define your stock symbol and API key
stock_symbol = "TCS"
news_api_key = '58827e1493184ade95c9370bc0732487'  # Replace with your actual News API key

# Set date range for the last 30 days
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# Fetch stock data using yfinance
symbol = "TCS.NS"  # NSE symbol for TCS
stock_data = yf.download(symbol, start=start_date, end=end_date, interval="5m")

if not stock_data.empty:
    stock_data.index = stock_data.index.tz_localize(None)
    stock_data.reset_index(inplace=True)
    stock_data.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    stock_data['DateTime'] = pd.to_datetime(stock_data['DateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
else:
    print("No stock data available for the specified date range.")

# Fetch news headlines from News API
def fetch_news_data(api_key, query, from_date):
    url = f'https://newsapi.org/v2/everything?q={query}&from={from_date}&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'articles' in data:
            articles = data['articles']
            headlines = [{'DateTime': article['publishedAt'], 'title': article['title']} for article in articles]
            return headlines
        else:
            print(f"Error: 'articles' key not found in response. Response: {data}")
            return []
    else:
        print(f"Error: Failed to fetch data. Status code: {response.status_code}")
        return []

# Perform sentiment analysis on news headlines
def analyze_sentiment(headlines):
    sentiments = []
    for headline in headlines:
        analysis = TextBlob(headline['title'])
        timestamp = pd.to_datetime(headline['DateTime']).strftime('%Y-%m-%d %H:%M:%S')
        sentiments.append({'DateTime': timestamp, 'title': headline['title'], 'sentiment': analysis.sentiment.polarity})
    return sentiments

# Main function to fetch, process, and visualize data
def main():
    # Fetch sentiment data
    headlines = fetch_news_data(news_api_key, stock_symbol, start_date)
    if headlines:
        sentiments = analyze_sentiment(headlines)
        sentiment_df = pd.DataFrame(sentiments)
    else:
        print("No sentiment data available.")
        return

    sentiment_df['DateTime'] = pd.to_datetime(sentiment_df['DateTime'])
    stock_data['DateTime'] = pd.to_datetime(stock_data['DateTime'])

    # Initialize sentiment score column in stock data with float type
    stock_data['Sentiment Score'] = 0.0
    for i in range(1, len(stock_data)):
        start_time = stock_data.loc[i-1, 'DateTime']
        end_time = stock_data.loc[i, 'DateTime']
        mask = (sentiment_df['DateTime'] > start_time) & (sentiment_df['DateTime'] <= end_time)
        sentiments_in_range = sentiment_df.loc[mask, 'sentiment']
        if not sentiments_in_range.empty:
            avg_sentiment = sentiments_in_range.mean()
            stock_data.loc[i, 'Sentiment Score'] = avg_sentiment

    stock_data.set_index('DateTime', inplace=True)
    dataset = stock_data[['Close', 'Sentiment Score']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Split data into 80% training and 20% testing
    train_data_len = int(len(scaled_data) * 0.80)
    train_data = scaled_data[:train_data_len]
    test_data = scaled_data[train_data_len - 60:]  # Include last 60 values for testing with 60 interval window

    # Prepare the training data with a 60-interval window
    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, :])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))

    # Build the LSTM model using an Input layer for shape specification
    model = Sequential([
        Input(shape=(X_train.shape[1], 2)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=30)

    X_test, y_test = [], dataset[train_data_len:]
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i, :])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 1))), axis=1))[:, 0]

    predicted_direction = np.where(predictions[1:] > predictions[:-1], 1, 0)
    actual_direction = np.where(y_test[1:, 0] > y_test[:-1, 0], 1, 0)
    accuracy = np.mean(predicted_direction == actual_direction)
    print(f"LSTM Model Accuracy: {accuracy * 100:.2f}%")

    mae = mean_absolute_error(y_test[:, 0], predictions)
    mse = mean_squared_error(y_test[:, 0], predictions)
    rmse = np.sqrt(mse)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Save results to Excel
    stock_data['Predicted Price'] = np.nan
    stock_data['Actual Price'] = np.nan
    stock_data['Predicted Direction'] = np.nan
    stock_data['Actual Direction'] = np.nan

    stock_data.loc[stock_data.index[train_data_len:], 'Predicted Price'] = predictions
    stock_data.loc[stock_data.index[train_data_len:], 'Actual Price'] = y_test[:, 0]
    stock_data.loc[stock_data.index[train_data_len + 1:], 'Predicted Direction'] = predicted_direction
    stock_data.loc[stock_data.index[train_data_len + 1:], 'Actual Direction'] = actual_direction

    metrics = {
        'Accuracy': accuracy * 100,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }

    with pd.ExcelWriter("tcs_stock_prediction_full_data.xlsx") as writer:
        stock_data.to_excel(writer, sheet_name="Data")
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="Metrics")

    print("\nResults saved to 'tcs_stock_prediction_full_data.xlsx'")

    plt.figure(figsize=(14, 6))
    plt.plot(stock_data.index[train_data_len:], y_test[:, 0], label='Actual Prices')
    plt.plot(stock_data.index[train_data_len:], predictions, label='Predicted Prices', color='red')
    plt.title('LSTM Model - Actual vs Predicted Prices for TCS')
    plt.xlabel('DateTime')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()