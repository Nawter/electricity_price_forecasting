# UK Imbalance Price Forecasting
Purpose of this repo is educational - the code is focused on simple & understandable Python (using only functions, not classes).

Project is split into:
-  elexton_data_scraping.py - pulls data from the Elexon API and saves in a sqlite database.
-  make_dataset.py - pulls data from the sqlite databse and creates a machine learning dataset
-  keras_models.py - contains functions to create feedforward & LSTM Keras models
-  feedforward.py - creates a Keras model using the saved ff_data dataset
-  lstm.py - creates a dataset & then a Keras model

Real data is included but it's unlikely you can train a very powerful model using only two features.  This is meant as a demonstration of how to implement timeseries forecasting in Python.

This project is built and maintained by Adam Green -  adam.green@adgefficiency.com.
