# -------------------------------------------------------------------------
# AUTHOR: Cristopher Hernandez
# FILENAME: decision_tree.py
# SPECIFICATION: Module which repeatedly trains a decision tree model with varying levels of training data.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import typing


class TestPrediction:
    """
    Helper class which tracks a prediction vs actual comparison.
    """
    def __init__(self, prediction, actual):
        self.prediction = prediction
        self.actual = actual

    @property
    def is_true_positive(self) -> bool:
        """
        Property determining whether this test is a True Positive.

        :return: A flag indicating whether this test is a True Positive.
        """
        return self.prediction == 1 and self.actual == 1

    @property
    def is_true_negative(self) -> bool:
        """
        Property determining whether this test is a True Negative.

        :return: A flag indicating whether this test is a True Negative.
        """
        return self.prediction == 0 and self.actual == 0

    @property
    def is_false_positive(self) -> bool:
        """
        Property determining whether this test is a False Positive.

        :return: A flag indicating whether this test is a False Positive.
        """
        return self.prediction == 1 and self.actual == 0

    @property
    def is_false_negative(self) -> bool:
        """
        Property determining whether this test is a False Negative.

        :return: A flag indicating whether this test is a False Negative.
        """
        return self.prediction == 0 and self.actual == 1


def get_accuracy(tests: typing.List[TestPrediction]) -> float:
    """
    Function which calculates accuracy using the given list of test predictions.

    :param tests: The test predictions to use to calculate accuracy.
    :return: The calculated accuracy.
    """
    true_positives = len(list(filter(lambda t: t.is_true_positive, tests)))
    true_negatives = len(list(filter(lambda t: t.is_true_negative, tests)))
    false_positives = len(list(filter(lambda t: t.is_false_positive, tests)))
    false_negatives = len(list(filter(lambda t: t.is_false_negative, tests)))
    return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)


def convert_data_set(data_frame: pd.DataFrame) -> np.ndarray:
    """
    Converts the given data frame into a numpy array, with all features transformed accordingly.
    Specifically:

    - Refund is encoded as 1 or 0, for yes or no respectively.
    - Marital Status is one-hot encoded.
    - Taxable income is expanded from the string format 'XK' to a full float number.

    :param data_frame: The data frame to convert.
    :return: The features transformed as a numpy array.
    """
    encoded_refund = data_frame['Refund'].replace({'Yes': '1', 'No': '0'}).astype(int)
    marital_status = pd.get_dummies(data_frame['Marital Status']).astype(int)
    taxable_income = data_frame['Taxable Income'].replace(to_replace='k', value='e3', regex=True).astype(float)
    training_data = pd.concat([encoded_refund, marital_status, taxable_income], axis=1)
    return training_data.to_numpy()


def convert_class_labels(data_frame: pd.DataFrame) -> np.ndarray:
    """
    Generates a Numpy array for the class label "Cheat" (encoded as 1 and 0, for 'yes' and 'no') in the given dataframe.

    :param data_frame: The dataframe to extract the class labels from.
    :return: The class labels encoded as a Numpy array.
    """
    cheat_labels = data_frame['Cheat'].replace({'Yes': '1', 'No': '0'}).astype(int)
    return cheat_labels.to_numpy()


data_sets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']
tree_image_dir = pathlib.Path(__file__).parent / 'images'
tree_image_dir.mkdir(parents=True, exist_ok=True)

for ds in data_sets:

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)

    # transform the original training features to numbers
    X = convert_data_set(df)

    # transform the original training classes to numbers
    Y = convert_class_labels(df)

    accuracy_tests = []

    # loop your training and test tasks 10 times here
    for i in range (10):

       # fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       # plotting the decision tree
       tree.plot_tree(
           clf,
           feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'],
           class_names=['Yes','No'],
           filled=True,
           rounded=True
       )
       plt.savefig(str(tree_image_dir / pathlib.Path(f'{ds}.{i}.png')))

       # read the test data and add this data to data_test NumPy
       data_test = pd.read_csv('cheat_test.csv', sep=',', header=0)
       test_X = convert_data_set(data_test)
       test_Y = convert_class_labels(data_test)
       test_predictions = []

       index: int
       data: np.ndarray
       for index, data in enumerate(test_X):
           # Generate predictions and track them to calculate accuracy
           class_predicted = clf.predict(data.reshape(1, -1))[0]
           true_label = test_Y[index]
           test_predictions.append(TestPrediction(class_predicted, true_label))

       # find the average accuracy of this model during the 10 runs (training and test set)
       accuracy_tests.append(get_accuracy(test_predictions))

    accuracy_average = sum(accuracy_tests) / len(accuracy_tests)

    # print the accuracy of this model during the 10 runs (training and test set).
    print(f'Final accuracy when training on {ds} is: {accuracy_average}')
print(f'See {tree_image_dir} for decision tree plot images')
