import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from constants import FILEPATH, FILENAME
from utils import split_train_test, get_feature_importance

# Load in the data
df = pd.read_csv(os.path.join(FILEPATH, FILENAME))

# Split into train and test sections
X_train, X_test, y_train, y_test = split_train_test(df)

# Fit a model on the train section
# Improve using a wider grid of hyperparams - just an example here.
clf = RandomForestClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
importances = clf.feature_importances_

# Calculate feature importance in random forest
feature_df = get_feature_importance(importances, df)
