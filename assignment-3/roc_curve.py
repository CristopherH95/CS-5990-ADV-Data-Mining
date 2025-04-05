# -------------------------------------------------------------------------
# AUTHOR: Cristopher Hernandez
# FILENAME: roc_curve.py
# SPECIFICATION: Generates a ROC curve for a decision tree classifier.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 30 minutes
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd


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


# read the dataset cheat_data.csv and prepare the data_training numpy array
df = pd.read_csv('cheat_data.csv', sep=',', header=0)

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
X = convert_data_set(df)

# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
y = convert_class_labels(df)

# split into train/test sets using 30% for test
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3)

# generate random thresholds for a no-skill prediction (random classifier)
ns_probs = np.random.uniform(low=0, high=1, size=(len(testX), 1))

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
positive_class_index = np.argwhere(clf.classes_ == 1)[0][0]
dt_probs = dt_probs[:, positive_class_index]

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % ns_auc)
print('Decision Tree: ROC AUC=%.3f' % dt_auc)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()
