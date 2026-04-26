from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SequentialFeatureSelector
from functools import partial

import pandas as pd
import numpy as np
import random as random

# Please do not change these random seeds.
np.random.seed(231)
random.seed(231)
model = KNeighborsClassifier(n_neighbors=5)

# --------------------------- Part 1: Load the Data ---------------------------

data_train = pd.read_csv('Training.csv')
X_train, y_train = data_train.drop('Status', axis=1), data_train['Status']

data_test = pd.read_csv('Test.csv')
X_test, y_test = data_test.drop('Status', axis=1), data_test['Status']

# --------------------------- Part 1: Data Preprocessing ---------------------------

# Identify numerical and categorical columns
numerical_cols = []  # List of numerical feature names
categorical_cols = []  # List of categorical feature names

### Step 1: Handle Missing Values ###
# Define and apply appropriate imputers for numerical and categorical features.
# Ensure consistency between X_train and X_test.

### Step 2: Encoding Categorical Features ###
# Convert categorical features into numerical format using encoding techniques.

### Step 3: Feature Scaling ###
# Apply scaling/normalization to features.

X_train_process = []  # Processed/transformed training set
X_test_process = []  # Processed/transformed test set

# Train the model using all available features.
model.fit(X_train_process, y_train)
y_pred = model.predict(X_test_process)
print(f"All Features: {balanced_accuracy_score(y_test, y_pred):.4f}")

# --------------------------- Part 2: Feature Ranking ---------------------------
# Use SelectKBest with mutual_info_classif to select the top 7 features from the processed training set X_train_process.
# You should use mutual_info_fix as the parameter for SelectKBest to ensure random_state is set to 231 for reproducibility.
mutual_info_fix = partial(mutual_info_classif, random_state=231)
selector_kbest = []  # Replace [] with SelectKBest implementation
X_train_kbest = []  # Replace [] with transformed training data
X_test_kbest = []  # Replace [] with transformed test data

# Train the model with selected features.
model.fit(X_train_kbest, y_train)
y_pred = model.predict(X_test_kbest)
print(f"Feature Ranking: {balanced_accuracy_score(y_test, y_pred):.4f}")

# --------------------------- Part 3: Sequential Feature Selection ---------------------------
# Use Sequential Backward Selection (SBS) to select a subset of 7 features from the processed training set X_train_process.
selector_sequential = []  # Replace [] with SequentialFeatureSelector implementation
X_train_sbfs = []  # Replace [] with transformed training data
X_test_sbfs = []  # Replace [] with transformed test data

# Train the model with selected features.
model.fit(X_train_sbfs, y_train)
y_pred = model.predict(X_test_sbfs)
print(f"Sequential Feature Selection: {balanced_accuracy_score(y_test, y_pred):.4f}")
