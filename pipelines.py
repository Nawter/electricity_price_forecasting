import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
            shifted = [x.shift(-abs(o), axis=0) for o in self.offsets]
            return pd.concat(shifted, axis=1)

if __name__ == '__main__':
    arr = np.arange(6)
    df = pd.DataFrame(arr.reshape(-1, 2))

    shifter = OffsetGenerator(mode='lag', offsets=[1, 2])

    shifted = shifter.transform(df)

    print(shifted)



