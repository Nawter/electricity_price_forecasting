import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pipelines import AsMatrix, ColumnSelector


raw_data = pd.read_csv('./elexon_data/elexon_report_data.csv',
                       index_col=0,
                       parse_dates=True)

f = 'imbalanceQuantityMAW'

train, test = train_test_split(raw_data, test_size=0.3)
train_index, test_index = train.index, test.index

make_target = make_pipeline(ColumnSelector('imbalancePriceAmountGBP'),
                            AsMatrix(),
                            StandardScaler())

y_train = make_target.fit_transform(train)
y_test = make_target.transform(test)

make_target_features = make_pipeline(ColumnSelector('imbalancePriceAmountGBP'))
