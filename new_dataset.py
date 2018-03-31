import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pipelines import *


raw_data = pd.read_csv('./elexon_data/elexon_report_data.csv',
                       index_col=0,
                       parse_dates=True)


train, test = train_test_split(raw_data, test_size=0.3)
train_index, test_index = train.index, test.index

HORIZIONS = [0, 1, 2, 3, 4, 10]
LAGS = [1, 2, 3, 4, 10]

make_target = make_pipeline(ColumnSelector('imbalancePriceAmountGBP'),
                            OffsetGenerator('horizion', HORIZIONS),
                            AlignPandas(max(LAGS), max(HORIZIONS)),
                            AsMatrix(),
                            StandardScaler())

y_train = make_target.fit_transform(train)
y_test = make_target.transform(test)

make_features = make_pipeline(ColumnSelector(['imbalancePriceAmountGBP',
                                              'imbalanceQuantityMAW']),
                              OffsetGenerator('lag', LAGS),
                              AlignPandas(max(LAGS), max(HORIZIONS)),
                              AsMatrix(),
                              StandardScaler())

x_train = make_features.fit_transform(train)
x_test = make_features.transform(test)






