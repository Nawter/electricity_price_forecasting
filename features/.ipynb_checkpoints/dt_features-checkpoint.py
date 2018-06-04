"""
A pipeline to make datetime features

https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
"""
import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from utils.pipelines import HalfHourlyCyclicalFeatures

if __name__ == '__main__':

    test_index = pd.DatetimeIndex(start='01-01-2016 00:00:00',
                                  end='01-02-2016 23:30:00',
                                  freq='30min')

    df = pd.DataFrame(index=test_index)

    pipe = make_pipeline(HalfHourlyCyclicalFeatures())

    out = pipe.fit_transform(df)

    import matplotlib.pyplot as plt

    f, a = plt.subplots()
    out.plot(ax=a)
    f.savefig('test.png')

    from utils.pd_functions import print_duplicates

    print_duplicates(out)


