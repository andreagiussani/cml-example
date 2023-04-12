from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from constants import (
    QUALITY_COLNAME,
    AGG_QUALITY_COLNAME,
    IMPORTANCE_COLNAME,
    FEATURE_COLNAME,
)


def split_train_test(df: pd.DataFrame) -> Union[pd.DataFrame]:
    """
    Split into train and test sets.
    This method also returns an aggregated version of the quality column.
    :param df: a pandas DataFrame
    :return: four pandas dataFrame denoting X_train, X_test, y_train, y_test
    """
    df[AGG_QUALITY_COLNAME] = df[QUALITY_COLNAME].apply(
        lambda x: 1 if x > 6 else 0
    )
    X = df.drop([QUALITY_COLNAME, AGG_QUALITY_COLNAME], axis=1)
    y = df[AGG_QUALITY_COLNAME]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def get_feature_importance(feat_importance: np.array, df: pd.DataFrame) -> pd.DataFrame:
    """
    This method returns the feature importance in a DataFrame, sorted by importance.
    :param feat_importance: np.array containing a list of floats
    :param df: a pandas DataFrame containing the raw training data
    :return: a pandas dataFrane containing the Feature Importance.
    """
    feature_df = pd.DataFrame(
        list(zip(df.columns, feat_importance)),
        columns=[FEATURE_COLNAME, IMPORTANCE_COLNAME]
    )
    feature_df = feature_df.sort_values(by=IMPORTANCE_COLNAME, ascending=False)
    return feature_df
