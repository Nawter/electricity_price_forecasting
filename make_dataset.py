import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pipelines as p

raw_data = pd.read_csv('./data/clean.csv',
                       index_col=0,
                       parse_dates=True)

train, test = train_test_split(raw_data, test_size=0.3)
train_index, test_index = train.index, test.index

HORIZIONS = [0, 1, 2, 3, 4, 10]
LAGS = [1, 2, 3, 4, 10]

make_target = make_pipeline(p.ColumnSelector('ImbalancePrice_excess_balance_[£/MWh]'),
                            p.OffsetGenerator('horizion', HORIZIONS),
                            p.AlignPandas(LAGS, HORIZIONS),
                            p.AsMatrix(),
                            StandardScaler())

y_train = make_target.fit_transform(train)
y_test = make_target.transform(test)

make_features = make_pipeline(p.OffsetGenerator('lag', LAGS),
                              p.AlignPandas(LAGS, HORIZIONS),
                              p.AsMatrix(),
                              StandardScaler())

x_train = make_features.fit_transform(train)
x_test = make_features.transform(test)

dataset = {'x_train': x_train,
           'y_train': y_train,
           'x_test': x_test,
           'y_test': y_test}

def print_dataset(dataset):
    for k, v in dataset.items():
        print('{}_{}'.format(k, v.shape))

print_dataset(dataset)
