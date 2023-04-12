import unittest
import pandas as pd
import numpy as np

from constants import (
    FIXED_ACIDITY,
    SULPHATES_COLNAME,
    ALCOHOL_COLNAME,
    PH_COLNAME,
    QUALITY_COLNAME,
)
from tests.fixtures import get_mocked_data
from utils import (
    split_train_test,
    get_feature_importance,
)


class WineQualityUtilsTestCase(unittest.TestCase):

    def setUp(self):
        self.columns = [
            FIXED_ACIDITY, PH_COLNAME, SULPHATES_COLNAME,
            ALCOHOL_COLNAME, QUALITY_COLNAME
        ]
        self.raw_data = pd.DataFrame(
            get_mocked_data(),
            columns=self.columns
        )

        self.feature_importance = np.array(
            [0.05743443, 0.03615136, 0.13846394, 0.251952]
        )

    def test__load_dataframe(self):
        df = self.raw_data
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 10)
        self.assertEqual(df.shape[1], 5)
        self.assertListEqual(list(df), self.columns)

    def test__split_train_test(self):
        X_train, X_test, y_train, y_test = split_train_test(self.raw_data)
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertEqual(X_train.shape[0], 8)
        self.assertEqual(X_test.shape[0], 2)
        self.assertEqual(X_train.shape[1], 4)
        self.assertListEqual(
            list(X_train),
            [FIXED_ACIDITY, PH_COLNAME, SULPHATES_COLNAME, ALCOHOL_COLNAME]
        )
        self.assertEqual(set(y_train.unique().tolist()), set([0, 1]))

    def test__get_feature_importance(self):
        feature_importance_df = get_feature_importance(
            self.feature_importance, self.raw_data
        )
        self.assertEqual(feature_importance_df.shape[0], 4)
