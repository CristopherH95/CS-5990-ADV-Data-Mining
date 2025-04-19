#-------------------------------------------------------------------------
# AUTHOR: Cristopher Hernandez
# FILENAME: knn.py
# SPECIFICATION: Tests various hyperparameter values for KNN, applied to weather data.
# FOR: CS 5990- Assignment #4
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#importing some Python libraries
import pandas as pd
import numpy as np
import math
import typing

from sklearn.neighbors import KNeighborsClassifier


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

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
X_training, y_training = read_weather_dataset("weather_training.csv")
#reading the test data
X_test, y_test = read_weather_dataset("weather_test.csv")
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
best_accuracy = -math.inf

#loop over the hyperparameter values (k, p, and w) ok KNN
for k_param in k_values:
    for p_param in p_values:
        for w_param in w_values:

            #fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=k_param, p=p_param, weights=w_param)
            clf = clf.fit(X_training, y_training)
            prediction_results = []

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            for x_test_sample, y_test_sample in zip(X_test, y_test):
                prediction = clf.predict(x_test_sample.reshape(1, -1))
                prediction_value, real_value = prediction[0], y_test_sample
                difference = 100 * (abs(prediction_value - real_value) / real_value)
                if difference < 15:
                    prediction_results.append(1)
                else:
                    prediction_results.append(0)

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            accuracy = sum(prediction_results) / len(prediction_results)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"Highest KNN accuracy so far: {accuracy}, Parameters: k={k_param}, p={p_param}, w={w_param}")
