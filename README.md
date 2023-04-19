Bitcoin Price Prediction using LSTM

This repository contains the implementation of a Long Short-Term Memory (LSTM) model for predicting the future price of Bitcoin. The model is trained on historical daily price data and predicts the next n days of prices along with upper and lower prediction intervals.

Workflow

Import required libraries: Import necessary libraries and fetch the dataset from CoinGecko API.
Data preprocessing: Preprocess the dataset, convert the timestamp to datetime, and split it into training and testing sets.
Model building and training: Create an ensemble of Sequential models with LSTM, Dropout, and Dense layers. Compile and train the models on the training and testing sets.
Evaluation: Calculate root mean squared error (RMSE) for the models and find the best model based on the lowest RMSE.
Prediction: Predict future prices using the ensemble of LSTM models, create prediction intervals, and visualize the predictions.

Data Preprocessing

Fetch the dataset from CoinGecko API and load it into a pandas DataFrame.
Convert the timestamp to datetime format and set it as the index.
Normalize the data using MinMaxScaler and split it into training and testing sets.

Model Building and Training

Build an ensemble of Sequential models with LSTM, Dropout, and Dense layers.
Compile the models with the mean squared error loss function and Adam optimizer.
Train the models on the training set and evaluate them on the testing set using RMSE.
Determine the best model based on the lowest test RMSE.

Evaluation

Calculate the RMSE for each model on the training and testing sets.
Compare the test RMSE values for different lookback values using a bar chart.
Select the best model based on the lowest test RMSE.

Prediction and Visualization

Predict future prices using the ensemble of LSTM models and the best lookback value.
Calculate upper and lower prediction intervals using the residuals from the training data.
Visualize the future price predictions along with the prediction intervals using a line chart.
Plot the actual vs. predicted Bitcoin prices for the test set and compare them with the future price predictions.
