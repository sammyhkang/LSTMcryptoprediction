Bitcoin Price Prediction using LSTM
This repository contains the implementation of a Long Short-Term Memory (LSTM) model for predicting the future price of Bitcoin. The model is trained on historical daily price data and predicts the next n days of prices along with upper and lower prediction intervals.

Table of Contents
Requirements and Installation
Dataset
Model Architecture
Training Process
Evaluation Metrics
Visualizations
Usage
Requirements and Installation
The code is written in Python 3.8 and requires the following libraries:

pandas
numpy
matplotlib
scikit-learn
tensorflow
keras
To install the required libraries, run the following command:

Copy code
pip install -r requirements.txt
Dataset
The dataset is fetched from the CoinGecko API, which provides historical daily price data for Bitcoin. The dataset is then cleaned and prepared for further processing.

The data includes the following columns:

timestamp: The timestamp of each data point.
price: The daily closing price of Bitcoin in USD.
Model Architecture
The model architecture consists of the following layers:

An input LSTM layer with 64 units and return_sequences set to True.
A Dropout layer with a dropout rate of 0.2.
Another LSTM layer with 32 units and return_sequences set to False.
A Dropout layer with a dropout rate of 0.2.
A Dense output layer with 1 unit.
The model is compiled using the Adam optimizer and the mean squared error loss function.

Training Process
The training process involves the following steps:

Splitting the data into a training set (80% of the data) and a test set (20% of the data).
Scaling the data using MinMaxScaler.
Determining the optimal lookback value based on the test RMSE.
Creating an ensemble of LSTM models by training multiple models and averaging their predictions.
Calculating the upper and lower prediction intervals using the residuals from the training data.
Evaluation Metrics
The performance of the model is evaluated using the root mean squared error (RMSE) metric. The RMSE is calculated for both the training and test sets.

Visualizations
The following visualizations are provided:

Comparison of RMSE values for different lookback values.
Actual vs. predicted Bitcoin prices along with prediction intervals for the test set.
Future Bitcoin price predictions with prediction intervals.
Usage
To train and evaluate the model, run the following command:

Copy code
python bitcoin_price_prediction.py
The output will display the RMSE for the training and test sets, as well as the visualizations mentioned above.

To predict future prices, update the n_future variable in the bitcoin_price_prediction.py script to the desired number of days and re-run the script. The predictions will be displayed as a plot along with the upper and lower prediction intervals.
