"""
A collection of generic sklearn pipelines
"""
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class AlignPandas(BaseEstimator, TransformerMixin):

    def __init__(self, LAGS, HORIZIONS):
        self.max_lag = max(LAGS)
        self.max_horizion = max(HORIZIONS)

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.iloc[self.max_lag:-self.max_horizion, :]


class AsMatrix(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        rows = x.shape[0]
        return x.as_matrix().reshape(rows, -1)


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.loc[:, self.columns]


class DropNans(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.dropna(axis=0)


class OffsetGenerator(BaseEstimator, TransformerMixin):
    """
    args
        mode (str) either lag or horizion
        offsets (list)
    """
    def __init__(self, mode, offsets):
        self.mode = mode
        self.offsets = offsets

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        """
        Not dropping na

        .shift(positive) -> lag
        .shift(negative) -> horizion
        """
        if self.mode == 'lag':
            shifted = [x.shift(abs(o), axis=0) for o in self.offsets]
            return pd.concat(shifted, axis=1)

        elif self.mode == 'horizion':
            shifted = [x.shift(-abs(o), axis=0) for o in self.offsets]
            return pd.concat(shifted, axis=1)

        else:
            raise ValueError('Mode of {} is not correct'.format(self.mode))