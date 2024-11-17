import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input, BatchNormalization
from keras.optimizers import RMSprop
from tensorflow.keras.losses import Huber
from sqlalchemy import create_engine
import finnhub

# PostgreSQL connection string
DATABASE_URI = 'postgresql://postgres:sree546372@localhost:5432/project_stock_data'

# Initialize API clients and keys
finnhub_client = finnhub.Client(api_key='cspo8a1r01qj9q8n4kpgcspo8a1r01qj9q8n4kq0')
fmp_api_key = 'Ia1k4CUzrWYOo9z1QXIJ0LrW3NP3wPKP'

# List of 100 stock tickers
top_100_tickers = [
    'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS',
    'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
    'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS',
    'HCLTECH.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS',
    'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS','TATACOMM.NS'
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS',
    'UPL.NS', 'WIPRO.NS', 'APOLLOHOSP.NS', 'DLF.NS', 'GAIL.NS', 'GLENMARK.NS',
    'GODREJCP.NS', 'HAVELLS.NS', 'HINDPETRO.NS', 'ICICIPRULI.NS', 'IOC.NS', 'IRCTC.NS',
    'LICHSGFIN.NS', 'LUPIN.NS', 'MFSL.NS', 'MOTHERSUMI.NS', 'ADANIPOWER.NS',
    'MRF.NS', 'MUTHOOTFIN.NS', 'NAUKRI.NS', 'PETRONET.NS', 'ZYDUSLIFE.NS', 'ADANIGREEN.NS', 
    'TRENT.NS', 'UBL.NS', 'UNIONBANK.NS', 'YESBANK.NS', 'ZYDUSWELL.NS', 'AARTIIND.NS',
    'ABFRL.NS', 'ADANIPOWER.NS', 'ALKEM.NS', 'ATGL.NS', 'BANDHANBNK.NS',
    'BHEL.NS', 'CANFINHOME.NS', 'COFORGE.NS', 'CUMMINSIND.NS', 'IDFC.NS', 'INDIAMART.NS',
    'INDUSTOWER.NS', 'IRFC.NS', 'J&KBANK.NS', 'JKCEMENT.NS', 'JSWENERGY.NS',
    'KAJARIACER.NS', 'KANSAINER.NS', 'KARURVYSYA.NS', 'KEI.NS', 'KPRMILL.NS',
    'LALPATHLAB.NS' ,'LICHSGFIN.NS', 'LUXIND.NS', 'METROPOLIS.NS',
    'MFSL.NS', 'MPHASIS.NS', 'UNIONBANK.NS', 'YESBANK.NS'
    ]

# Fetch data from FinancialModelingPrep
def fetch_fmp_data(symbol):
    url = f'https://financialmodelingprep.com/api/v3/ratios/{symbol}?apikey={fmp_api_key}'
    response = requests.get(url)
    data = response.json()
    return data[0] if data else {}

def fetch_combined_data(ticker):
    fmp_symbol = ticker.split('.')[0]
    usd_to_inr_rate = 83.0

    # Attempt to fetch Finnhub data
    try:
        finnhub_data = finnhub_client.company_basic_financials(fmp_symbol, 'all')
        finnhub_metrics = finnhub_data.get('metric', {})
    except:
        finnhub_metrics = {}

    # Fetch FMP data as fallback
    fmp_data = fetch_fmp_data(fmp_symbol)

    # Fetch Yahoo Finance data
    yf_data = yf.Ticker(ticker)
    yf_info = yf_data.info
    yf_cash_flow = yf_data.cashflow
    yf_balance_sheet = yf_data.balance_sheet

    # Convert FMP data from USD to INR if available
    def convert_to_inr(value):
        return value * usd_to_inr_rate if value else None

    # Combine data from all sources with prioritized fallbacks
    combined_data = {
        "Market Cap (INR)": finnhub_metrics.get('marketCapitalization') or convert_to_inr(fmp_data.get('marketCap')) or yf_info.get('marketCap'),
        "52-Week High (INR)": finnhub_metrics.get('52WeekHigh') or convert_to_inr(fmp_data.get('high')) or yf_info.get('fiftyTwoWeekHigh'),
        "52-Week Low (INR)": finnhub_metrics.get('52WeekLow') or convert_to_inr(fmp_data.get('low')) or yf_info.get('fiftyTwoWeekLow'),
        "Stock P/E (TTM)": finnhub_metrics.get('peBasicExclExtraTTM') or fmp_data.get('priceEarningsRatio') or yf_info.get('trailingPE'),
        "Dividend Yield (%)": finnhub_metrics.get('dividendYieldIndicatedAnnual') or fmp_data.get('dividendYield') or (yf_info.get('dividendYield') * 100 if yf_info.get('dividendYield') else None),
        "Book Value (INR)": finnhub_metrics.get('bookValuePerShareAnnual') or convert_to_inr(fmp_data.get('bookValuePerShare')) or yf_info.get('bookValue'),
        "Debt to Equity": finnhub_metrics.get('totalDebt/totalEquityAnnual') or fmp_data.get('debtEquityRatio') or yf_info.get('debtToEquity'),
        "Price to Book Value": finnhub_metrics.get('pbAnnual') or fmp_data.get('priceToBookRatio') or yf_info.get('priceToBook'),
        "Price to Sales": finnhub_metrics.get('psAnnual') or fmp_data.get('priceToSalesRatio') or yf_info.get('priceToSalesTrailing12Months'),
        "Sales Growth 3Yrs": finnhub_metrics.get('revenueGrowth3Y') or fmp_data.get('threeYearRevenueGrowth') or yf_info.get('threeYearAverageReturn'),
        "Sales Growth 5Yrs": finnhub_metrics.get('revenueGrowth5Y') or fmp_data.get('fiveYearRevenueGrowth') or yf_info.get('fiveYearAverageReturn'),
        "OPM 5Yrs": finnhub_metrics.get('grossMargin5Y') or (fmp_data.get('operatingMargin') * 100 if fmp_data.get('operatingMargin') else None) or (yf_info.get('operatingMargins') * 100 if yf_info.get('operatingMargins') else None),
        "NPM Last Year": finnhub_metrics.get('netProfitMarginAnnual') or (fmp_data.get('netProfitMargin') * 100 if fmp_data.get('netProfitMargin') else None) or (yf_info.get('profitMargins') * 100 if yf_info.get('profitMargins') else None),
        "Current Ratio": finnhub_metrics.get('currentRatioAnnual') or fmp_data.get('currentRatio') or yf_info.get('currentRatio'),
        "Free Cash Flow 3Yrs (INR)": convert_to_inr(fmp_data.get('freeCashFlow')) or finnhub_metrics.get('freeCashFlowAnnual3Y') or (yf_cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in yf_cash_flow.index else None),
        
        # Additional Yahoo Finance Metrics
        "Current Price (INR)": yf_info.get('currentPrice'),
        "ROE": yf_info.get('returnOnEquity') or fmp_data.get('roe'),
        "EPS (INR)": convert_to_inr(fmp_data.get('eps')) or yf_info.get('trailingEps'),
        "Total Debt (INR)": convert_to_inr(fmp_data.get('totalDebt')) if 'totalDebt' in fmp_data else (yf_balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in yf_balance_sheet.index else None),
        "Free Cash Flow (INR)": yf_cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in yf_cash_flow.index else None,
        "Reserves (INR)": yf_balance_sheet.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in yf_balance_sheet.index else None,
        "Operating Profit Margin (%)": yf_info.get('operatingMargins') * 100 if yf_info.get('operatingMargins') else None,
        "Net Profit Margin (%)": yf_info.get('profitMargins') * 100 if yf_info.get('profitMargins') else None,
        "PEG Ratio": yf_info.get('pegRatio')
    }

    # Remove None or 'N/A' values
    combined_data = {k: v for k, v in combined_data.items() if v is not None and v != 'N/A'}
    
    return combined_data

# Initialize an empty DataFrame to store data for all stocks
all_stock_data = pd.DataFrame()

# Loop over each stock in the top 200 list
for ticker in top_100_tickers:
    try:
        print(f"Processing {ticker}...")

        # Fetch historical stock data for the ticker
        df = yf.download(ticker, start='2012-01-01', end=datetime.now())
        
        # Skip the ticker if data is empty
        if df.empty:
            print(f"No data found for {ticker}. Skipping...")
            continue

        # Retrieve combined data from all APIs
        combined_metrics = fetch_combined_data(ticker)
        for key, value in combined_metrics.items():
            df[key] = value  # Add combined metrics as new columns

        # Select features for training and prediction
        df_lstm = df[['Close', 'Open', 'High', 'Low', 'Volume']]
        dataset = df_lstm.values

        # Scale the data to the range 0 to 1 for better performance with LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Split data into training and testing sets (95% training)
        train_data_len = int(np.ceil(len(scaled_data) * 0.95))
        train_data = scaled_data[:train_data_len]
        test_data = scaled_data[train_data_len:]

        # Prepare the training dataset
        x_train, y_train = [], []
        window_size = 120  # 120 days of historical data used for prediction
        for i in range(window_size, len(train_data)):
            x_train.append(train_data[i-window_size:i, :])
            y_train.append(train_data[i, 0])

        # Convert to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

        # Build the LSTM model
        model = Sequential([
            Input(shape=(x_train.shape[1], x_train.shape[2])),
            Bidirectional(LSTM(units=128, return_sequences=True)),
            Dropout(0.3),
            BatchNormalization(),
            Bidirectional(LSTM(units=64, return_sequences=True)),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(units=32, return_sequences=False),
            Dropout(0.3),
            Dense(50),
            Dense(1)
        ])

        # Compile and train the model
        optimizer = RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss=Huber())
        
        # Train the model on the training data
        model.fit(x_train, y_train, batch_size=32, epochs=30)

        # Step 2: Test the model on unseen test data
        x_test = []
        y_test = dataset[train_data_len:, 0]  # Unscaled actual Close values for test comparison

        test_scaled = scaled_data[train_data_len-window_size:]  # Include last 120 values from train
        for i in range(window_size, len(test_scaled)):
            x_test.append(test_scaled[i-window_size:i, :])

        # Convert x_test to a numpy array and reshape for LSTM input
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

        # Predict the prices for the test dataset
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(np.concatenate([predictions, np.zeros((predictions.shape[0], dataset.shape[1] - 1))], axis=1))[:, 0]

        # Add predicted values to the DataFrame
        df.loc[df.index[train_data_len:], 'Predicted'] = predictions

        # Final adjustments
        # Step 3: Predict future prices (next 1 days)
        future_days = 1
        last_window = scaled_data[-window_size:]  # Last window of data for prediction
        future_predictions = []

        for day in range(future_days):
            # Reshape last_window for LSTM input
            last_window_reshaped = np.reshape(last_window, (1, last_window.shape[0], last_window.shape[1]))

            # Predict the next day price
            next_day_pred = model.predict(last_window_reshaped)
            future_predictions.append(next_day_pred[0, 0])

            # Update the last_window with the predicted value
            new_row = np.concatenate([next_day_pred, np.zeros((1, dataset.shape[1] - 1))], axis=1)  # Predicted Close + dummy zeros for other features
            new_row_scaled = scaler.transform(new_row)  # Scale the predicted value

            # Update the window to include the new predicted row
            last_window = np.vstack([last_window[1:], new_row_scaled])

        # Inverse transform the future predictions to get back to the original scale
        future_predictions = scaler.inverse_transform(np.concatenate([np.array(future_predictions).reshape(-1, 1), np.zeros((future_days, dataset.shape[1] - 1))], axis=1))[:, 0]

        # Step 4: Predict whether the future price is a rise (1) or fall (0)
        future_rise_fall = [1 if future_predictions[0] > predictions[-1] else 0]
        future_rise_fall += [1 if future_predictions[i] > future_predictions[i-1] else 0 for i in range(1, future_days)]

        # Modify future predictions DataFrame to include Price_Difference and Percentage_Change
        today_prediction = predictions[-1]  # Today's predicted price (last prediction in test data)
        price_difference = future_predictions[0] - today_prediction
        percentage_change = (price_difference / today_prediction) * 100

        # Create a DataFrame for future predictions
        future_df = pd.DataFrame({
            'Date': pd.date_range(df.index[-1] + timedelta(1), periods=future_days),
            'Future_Price': future_predictions,
            'Future_Rise_Fall': future_rise_fall,
            'Price_Difference': [price_difference] + [np.nan] * (future_days - 1),
            'Percentage_Change': [percentage_change] + [np.nan] * (future_days - 1)
        })

        future_df.set_index('Date', inplace=True)

        # Append the future predictions DataFrame to the original DataFrame
        df = pd.concat([df, future_df])

        # Add Moving Averages (20, 50, 200), Previous Day Price, Change in Price, etc.
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Price_Percent_Change'] = (df['Price_Change'] / df['Prev_Close']) * 100
        df['Prev_Volume'] = df['Volume'].shift(1)
        df['Volume_Change'] = df['Volume'] - df['Prev_Volume']
        df['Volume_Percent_Change'] = (df['Volume_Change'] / df['Prev_Volume']) * 100

        # Step 5: Add the 'Ticker' column
        df['Ticker'] = ticker  # Add the ticker column

        # Append the data for this stock to the main DataFrame
        all_stock_data = pd.concat([all_stock_data, df])

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue  # Skip this ticker if any error occurs

# Save DataFrame with all stock data to PostgreSQL
engine = create_engine(DATABASE_URI)
all_stock_data.to_sql('top100_stock_predictions', engine, if_exists='replace', index=True)

print("Data successfully saved to PostgreSQL!")