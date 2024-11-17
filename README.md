# AI-ENHANCED STOCK MARKET INSIGHTS FOR INDIAN STOCKS USING TABLEAU

## Overview

This project develops a stock market analysis and prediction system that integrates machine learning, sentiment analysis, and data visualization to enhance predictive accuracy. Leveraging reliable data from Yahoo Finance and sentiment scores derived from News APIs, the system combines historical stock data with market sentiment. Three machine learning models - ARIMA, Prophet, and Long Short-Term Memory (LSTM) are employed, with LSTM selected and optimized for its ability to capture sequential patterns and non-linear trends. By automating data collection, preprocessing, and model training, the system ensures efficient data handling and management via PostgreSQL. Interactive dashboards are developed in Tableau and made publicly accessible through Tableau Public, providing real-time insights into stock performance. By enhancing predictive reliability and delivering actionable insights, the system addresses the complexities of stock market analysis and empowers investors with a scalable, user-friendly tool.

## Features

- **Real-Time Data Integration**: Fetches stock data from Yahoo Finance and integrates sentiment scores from news APIs.
- **Machine Learning Models**: Employs ARIMA, Prophet, and LSTM models to predict stock prices, with LSTM offering the highest accuracy.
- **Sentiment Analysis**: Combines stock data with sentiment analysis from news headlines for improved prediction.
- **Data Visualization**: Interactive dashboards created with Tableau Public, accessible for real-time analysis.
- **PostgreSQL Integration**: Efficient data storage and management for historical and predicted stock data.

## Project Workflow

1. **Data Collection**: Fetch stock and news data using APIs.
2. **Data Preprocessing**: Clean and normalize data for compatibility with machine learning models.
3. **Model Training and Evaluation**: Train multiple models (LSTM, ARIMA, Prophet) and select the best-performing one.
4. **Sentiment Integration**: Incorporate sentiment analysis for refined predictions.
5. **Data Storage**: Store processed data in PostgreSQL for easy retrieval.
6. **Visualization**: Create interactive dashboards in Tableau Public for insights.
7. **Automation Planning**: Explore GCP Cron Jobs for automating data collection and updates.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, Pandas, Numpy, Matplotlib, TextBlob
- **Machine Learning Models**: LSTM, ARIMA, Prophet
- **Data Storage**: PostgreSQL
- **Visualization**: Tableau Public
- **Cloud Services**: Google Cloud Platform (GCP)

## Getting Started

1. Clone this repository:
   ```bash
   git clone <[Your Repository URL](https://public.tableau.com/app/profile/sreekanth.r2017/viz/IndianStockMarketInsightsIntractiveVisualizations/StockAnalysis1)>
   cd <BITS-MTech-Final-Year-Project>
   ```

2. Set up your API keys for Yahoo Finance and News APIs in the configuration file.

3. Run the Python file.

4. Access the PostgreSQL database for processed data or view dashboards in Tableau Public 

## Results

- **Accuracy**: The LSTM model achieved a prediction accuracy of over 60%.

- **Future Enhancements**: Multi-year data integration and full pipeline automation using GCP Cron Jobs.

## How to Use the Dashboards

- Access the public Tableau dashboards at: [[Tableau Public Dashboard URL](https://public.tableau.com/app/profile/sreekanth.r2017/viz/IndianStockMarketInsightsIntractiveVisualizations/StockAnalysis1)]
- Explore stock trends, sentiment impact, and predictive insights through interactive visualizations.

## Limitations

- **News Data Restriction**: Sentiment data is limited to a 1-month period due to API constraints.
- **Automation**: Currently, automation workflows are under planning and not fully implemented.

## Future Work

- Incorporate multi-year datasets to improve accuracy.
- Fully automate workflows using GCP Cron Jobs for real-time updates.
- Expand stock coverage to include additional companies and indices.

## Contact

For any queries or contributions, feel free to contact:

- **Email**: [perfectvlogger333@gmail.com]
- **GitHub Repository**: [[Repository URL](https://github.com/sreekanth-rajan/BITS-MTech-Final-Year-Project)]

