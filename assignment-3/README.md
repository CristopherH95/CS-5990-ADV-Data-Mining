# Assignment 3: Decision Tree & ROC Curve

This assignment response implements two simple Python programs.

## Program 1: decision_tree.py

This program performs the following tasks:

1. Reads in provided `cheat_training_1.csv`, `cheat_training_2.csv`, `cheat_training_3.csv`, and `cheat_test.csv` files.
2. Converts and prepares training and test data, in order to train and test a decision tree model.
3. Trains a decision tree 10 times for each training data set, saving an image of the generated tree to an `images` folder.
4. Prints out the final average accuracy (from the 10 iterations) for each training data set to the console.

## Program 2: roc_curve.py

This program performs the following tasks:

1. Reads in provided `cheat_data.csv` file.
2. Converts and prepares training and test data, in order to train and test a decision tree model.
3. Trains a decision tree model.
4. Generates a series of "random threshold" values.
5. Generates a series of prediction probabilities on test samples, using the trained classifier.
6. Plots a ROC curve using the random threshold values and prediction probabilities for the positive class case.
