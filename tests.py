import pandas as pd
import numpy as np

from pipelines import OffsetGenerator

data = np.arange(10).reshape(-1, 2)
DATA = pd.DataFrame(data)


def test_simple_lag():
    offsetter = OffsetGenerator('lag', [1])

    lagged = offsetter.transform(DATA)

    assert pd.isnull(lagged.iloc[0, 1])
    assert lagged.iloc[2, 0] == 2
    assert lagged.iloc[4, 1] == 7


def test_complex_lag():
    offsetter = OffsetGenerator('lag', [2, 3])

    lagged = offsetter.transform(DATA)

    assert pd.isnull(lagged.iloc[0, 0])
    assert pd.isnull(lagged.iloc[1, 1])
    assert lagged.iloc[2, 0] == 0
    assert lagged.iloc[4, 1] == 5

    assert pd.isnull(lagged.iloc[2, 3])
    assert lagged.iloc[3, 3] == 1
    assert lagged.iloc[4, 2] == 2

def test_simple_horizion():
    offsetter = OffsetGenerator('horizion', [1])

    horizions = offsetter.transform(DATA)

    assert horizions.iloc[0, 0] == 2
    assert horizions.iloc[0, 1] == 3
    assert horizions.iloc[2, 0] == 6
    assert pd.isnull(horizions.iloc[4, 1])


def test_complex_horizion():
    offsetter = OffsetGenerator('horizion', [2, 3])

    horizions = offsetter.transform(DATA)

    assert horizions.iloc[0, 1] == 5
    assert horizions.iloc[1, 0] == 6
    assert pd.isnull(horizions.iloc[3, 1])
    assert pd.isnull(horizions.iloc[4, 0])

    assert horizions.iloc[0, 2] == 6
    assert horizions.iloc[1, 3] == 9
    assert pd.isnull(horizions.iloc[2, 3])
    assert pd.isnull(horizions.iloc[4, 2])
