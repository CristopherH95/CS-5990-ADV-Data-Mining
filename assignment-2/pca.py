# -------------------------------------------------------------------------
# AUTHOR: Cristopher Hernandez
# FILENAME: pca.py
# SPECIFICATION: Simple program using PCA to test removing features from a dataset
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# importing some Python libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv("heart_disease_dataset.csv")

# Create a training matrix without the target variable (Heart Disease)
df_features = df.copy()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

# Get the number of features
num_features = len(df_features.columns)
# Keep track of test results
experiments = []


# Run PCA for 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    reduced_data = df_features.drop(df_features.columns[i], axis=1)

    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)

    # Store PC1 variance and the feature removed
    experiments.append((pca.explained_variance_ratio_[0], df_features.columns[i]))

# Find the maximum PC1 variance
highest_pc1 = max(experiments, key=lambda x: x[0])

# Print results
print(f"Highest PC1 variance found: {highest_pc1[0]} when removing {highest_pc1[1]}")
