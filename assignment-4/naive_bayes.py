#-------------------------------------------------------------------------
# AUTHOR: Cristopher Hernandez
# FILENAME: naive_bayes.py
# SPECIFICATION: Tests various hyperparameter values for Naive Bayes, applied to weather data.
# FOR: CS 5990- Assignment #4
# TIME SPENT: 20 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
import pandas as pd
import numpy as np
import typing
import math

from sklearn.naive_bayes import GaussianNB


# noinspection DuplicatedCode
def read_weather_dataset(path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    dataset_frame = pd.read_csv(
        path,
        sep=',',
        header=0,
        parse_dates=["Formatted Date"],
        index_col=["Formatted Date"],
        dtype={
            "Humidity": np.float64,
            "Wind Speed (km/h)": np.float64,
            "Wind Bearing (degrees)": np.float64,
            "Visibility (km)": np.float64,
            "Pressure (millibars)": np.float64,
            "Temperature (C)": np.float64
        },
    )
    # 1 additional class was added from the provided template.
    # this is because the previous ceiling of 39 cuts off the maximum value in the training dataset.
    bins = list(range(-22, 45, 6))
    dataset_frame["Class"] = pd.cut(
        dataset_frame["Temperature (C)"],
        bins=bins,
        labels=False,
        include_lowest=True,
        right=True,
    )
    return (
        dataset_frame[
            ["Humidity", "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)"]
        ].to_numpy(),
        dataset_frame["Class"].to_numpy()
    )

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
X_training, y_training = read_weather_dataset("weather_training.csv")

#reading the test data
X_test, y_test = read_weather_dataset("weather_test.csv")

best_accuracy = -math.inf

#loop over the hyperparameter value (s)
for s_param in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s_param)
    # noinspection DuplicatedCode
    clf = clf.fit(X_training, y_training)
    prediction_results = []

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    for x_test_sample, y_test_sample in zip(X_test, y_test):
        prediction = clf.predict(x_test_sample.reshape(1, -1))
        prediction_value, real_value = prediction[0], y_test_sample
        difference = 100 * (abs(prediction_value - real_value) / real_value)
        if difference < 15:
            prediction_results.append(1)
        else:
            prediction_results.append(0)

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    accuracy = sum(prediction_results) / len(prediction_results)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f"Highest NB accuracy so far: {accuracy}, Parameters: s={s_param}")
