This Jupyter Notebook provides a comprehensive and detailed script that predicts cryptocurrency prices using a deep learning architecture based on LSTM (Long Short-Term Memory) and Dense layers for time series forecasting. The goal of this notebook is to assist users in making informed investment decisions and assessing risk by inputting historical price data and predicting future prices.

The main sections of the notebook are as follows:

A) Import libraries and modules: The required libraries and modules are imported to provide the necessary functions and tools for the implementation. Some of the key imports include pandas, NumPy, Keras, and scikit-learn.

B) Data fetching and preprocessing:

1. fetch_data(ticker, interval): Fetch historical price data from the CoinGecko API based on the user's input for the cryptocurrency ticker and interval (1 hour, 4   hours, or 24 hours).

2. normalize_data(data): Normalize the fetched data using the MinMaxScaler from scikit-learn to ensure that the model can efficiently process the input data.

3. split_data(scaled_data): Split the normalized data into train and test sets, with 80% of the data used for training and 20% for testing.
  
C) Creating datasets:

1. create_dataset(dataset, look_back): Create input and output datasets for the LSTM model by generating sequences of data points (with a given look-back value) and their corresponding future values.

D) Finding the best look-back value:

1. find_best_lookback(train, test, lookback_values, scaler): Iterate through a range of look-back values, train the model for each value, and calculate the test Root Mean Squared Error (RMSE). The best look-back value corresponds to the lowest test RMSE.

E) Model building and training:

1. build_and_train_model(train_X, train_y, test_X, test_y, look_back, dropout_rate, learning_rate, optimizer, bidirectional): Build an LSTM model with various hyperparameters, including dropout rate, learning rate, optimizer, and bidirectional LSTM layers. Train the model on the historical price data and evaluate its performance on the test set.

F) Model evaluation and selection:

1. The script iterates through different combinations of hyperparameters (dropout rates, learning rates, optimizers, and bidirectional options) and stores the models with their respective validation losses.
2. It then selects the best model based on the lowest validation loss.

G) Price prediction and confidence intervals:

1. calculate_rmse(train_y, test_y, trainPredict, testPredict, scaler): Calculate the train and test RMSE for the best model.
2. calculate_intervals(trainPredict, testPredict, train_data, scaler, look_back): Calculate the confidence intervals for the price predictions using the residuals from the training data.
3. predict_future_prices(models, data, look_back, n_future): Predict future prices using the trained model and input data.

H) Visualizing the results:

1. plot_future_prices(data, future_dates, future_prices, bounds1, bounds2, interval): Plot the actual and predicted prices, along with their confidence intervals, to help users visualize the model's performance and the predicted prices.

To use this notebook, you will need to input the cryptocurrency ticker (e.g., 'bitcoin' or 'ethereum') and the desired interval (1 for 1 hour, 4 for 4 hours, or 24 for 1 day). The script will then fetch historical price data, preprocess it, build and train an LSTM model, and predict future prices. You can visualize the results in the form
