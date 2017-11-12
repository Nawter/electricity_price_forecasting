import os
import sqlite3

import pandas as pd
import sklearn.preprocessing


def query_sql(db_path, table_name):
    """
    Uses pandas to pull data from our sqlite database

    args
        db_path (str) : location of the database
        table_name (str) : name of the table we want

    returns
        data (pd.DataFrame) :
    """
    print('Pulling data for table {} from {}'.format(db_path, table_name))
    #  connect to our database
    conn = sqlite3.connect(db_path)
    #  pull data by selecting the entire table
    data = pd.read_sql(sql='SELECT * from '+str(table_name), con=conn)
    data.set_index('index', drop=True, inplace=True)
    #  close the connection
    conn.close()
    return data


def offset_generator(series, offsets):
    """
    Generates a dataframe containing offsets from the series

    Can be used for generating either
        lags = positive offsets
        horizions = negative offsets

    """
    print('creating offsets for {}'.format(series.name))
    #  first we apply the conversion

    output = pd.DataFrame()
    for off in offsets:
        output = pd.concat([output, series.shift(-off)], axis=1).dropna()

    output.columns = ['{} offset {} HH'.format(series.name, offset) for offset in offsets]
    return output


def make_test_train(features, target, split=0.3):
    """
    Splits our features & target into test & training data.

    args
        features
        target

    returns
        x_train
        x_test
        y_train
        y_test
    """
    def split_test_train(df, split):
        """
        Helper function to do the splitting.
        """
        train = df.iloc[:split, :]
        test = df.iloc[split:, :]
        return train, test

    #  first we align our features and target
    #  i.e. make sure they have the same indicies
    features, target = features.align(target, axis=0, join='inner')
    assert features.shape[0] == target.shape[0]

    print('split data into test & train with ratio {}'.format(split))
    split = int(features.shape[0] * (1 - split))
    x_train, x_test = split_test_train(features, split)
    y_train, y_test = split_test_train(target, split)

    assert x_train.index[0] == y_train.index[0]
    assert x_train.index[-1] == y_train.index[-1]
    assert x_test.index[0] == y_test.index[0]
    assert x_test.index[-1] == y_test.index[-1]

    print('training period from {} to {}'.format(x_train.index[0], x_train.index[-1]))
    print('test period from {} to {}'.format(x_test.index[0], x_test.index[-1]))

    return x_train, x_test, y_train, y_test


def scale_train_test(train, test):
    """
    Scales each column of a DataFrame

    Fits a sklearn scaler to the training data
    Then apply this scaler to the test data

    args
        train (pd.DataFrame)
        test (pd.DataFrame)

    returns
        train_scaled (pd.DataFrame) : transformed training data
        scalers (list) : a list of sklearn scalers (fitted on the training data)
        train_scaled (pd.DataFrame) : transformed test data
    """
    def scale_series(series):
        """
        scales a single series - returns the scaled series and the scaler
        """
        assert len(series.shape) == 1, 'input has too many dimensions'
        #  make a scaler object
        sclr = sklearn.preprocessing.StandardScaler()
        #  fit it and transform the data (ie scale the data)
        scaled = sclr.fit_transform(series.values.reshape(-1, 1))
        #  flatten the series
        scaled = scaled.reshape(-1)
        #  turn back into a series object
        scaled = pd.Series(data=scaled,
                           index=series.index,
                           name=series.name)
        return scaled, sclr

    assert train.shape[1] == test.shape[1]
    #  empty lists to hold our output data
    train_scaled_list, scalers, test_scaled_list = [], [], []
    #  use iteritems to iterate over column names and columns
    for column, train_series in train.iteritems():
        train_scaled, scaler = scale_series(train_series)
        train_scaled_list.append(train_scaled)
        scalers.append(scaler)

    assert train.shape[1] == len(scalers)

    for scaler, (column, test_series) in zip(scalers, test.iteritems()):
        test_data = test_series.values.reshape(-1, 1)
        idx = test_series.index
        test_scaled = scaler.transform(test_data).reshape(-1)
        test_scaled = pd.Series(data=test_scaled,
                                index=idx,
                                name=column)
        test_scaled_list.append(test_scaled)

    train_scaled = pd.concat(train_scaled_list, axis=1)
    test_scaled = pd.concat(test_scaled_list, axis=1)

    return train_scaled, scalers, test_scaled


def make_datetime_features(df):
    """
    Adds datetime features to a pandas DataFrame

    args
        df (pd.DataFrame) : df to add datetime features to

    Returns
        df (pd.DataFrame) : df with datetime features
        dummies () : the created datetime features
    """
    #  first we make our index a date time index
    df.index = pd.to_datetime(df.index)
    index = df.index
    #  make some datetime features
    #  probably not the most efficient way to do it - but easy to understand
    print('making date time features')
    month = [idx.month for idx in index]
    day = [idx.day for idx in index]
    hour = [idx.hour for idx in index]
    minute = [idx.minute for idx in index]
    weekday = [idx.weekday() for idx in index]

    #  turn the datetime features into dummies
    features = [day, hour, minute, weekday]
    feature_names = ['day', 'hour', 'minute', 'weekday']

    #  loop over the created dummies
    dummies = []
    for feature, name in zip(features, feature_names):
        dummy = pd.get_dummies(feature)
        dummy.columns = ['D_' + name + '_' +str(col) for col in dummy.columns]
        dummy.index = index
        dummies.append(dummy)

    #  join all our dummy variables together
    dummies = pd.concat(dummies, axis=1)
    #  join our dummy variables onto the dataframe
    df = pd.concat([df, dummies], axis=1)
    return df, dummies


if __name__ == '__main__':
    """
    Code below is an example of how to make a dataset for use in a feedforward
    neural network
    """
    DATABASE_NAME = 'elexon_data/ELEXON_DATA.sqlite'
    TARGET = 'imbalancePriceAmountGBP'
    data_dicts = [{'report':'B1770', 'cols':['imbalancePriceAmountGBP']},
                  {'report':'B1780', 'cols':['imbalanceQuantityMAW']}]

    data = []
    for data_dict in data_dicts:
        #  first we unpack our dictionary
        report_name = data_dict['report']
        cols = data_dict['cols']

        #  grab the data from the sqlite database
        raw_data = query_sql(db_path=DATABASE_NAME, table_name=report_name)
        data.append(raw_data.loc[:, cols])

    #  join our list of DataFrames into a single dataframe
    #  axis=1 joins them side by side
    data = pd.concat(data, axis=1)

    """
    We now have the data we want from our SQL database

    Now we can start to process this dataset to train a neural network with
    - extract the taget and features
    - created lagged features and horizion targets
    - split into test & train_index
    - scale / normalize / standardize
    - add datetime features

    Our first step is to split into test & train
    We do this step first so that we can fit our scalers to our training data
     and use these scalers to transform our test data
    """

    """
    First we extract the features & target
    """
    target = data.loc[:, TARGET]
    features = data.drop(TARGET, axis=1)

    """
    Now we generate lagged features
    """
    #  create lists of the lags we want to include
    #  note that for lags we want to start at 1
    #  this is so that we don't include the current price (indexed at 0)
    hh_lags = [-i for i in range(1,5)]
    hh_horizions = [i for i in range(0, 8)]

    #  now use the offset_generator function to create the lagged dataframes
    #  for the features I use a list comprehension to iterate over each column in
    #   the features dataframe
    feature_lagged = [offset_generator(series, hh_lags) for _, series in features.iteritems()]

    #  creating lagged features from our target
    target_lagged = offset_generator(target, hh_lags)

    #  append the features generated from the target onto the lagged features
    feature_lagged.append(target_lagged)
    #  concat all the features dataframes together
    features = pd.concat(feature_lagged, axis=1)

    #  finally create our target horizions (our forecast)
    target = offset_generator(target, hh_horizions)

    """
    Now we have generated lagged features and made horizions for our targets
    We can now split into test & train data
    """
    x_train, x_test, y_train, y_test = make_test_train(features, target)

    """
    Now we can fit & transform our training data and transform our test data
    """

    y_train, y_scalers, y_test = scale_train_test(y_train, y_test)
    x_train, x_scalers, x_test = scale_train_test(x_train, x_test)

    """
    Now we have our scaled training & test data
    We can add datetime features to our features (x_train & x_test)
    """

    x_train, _ = make_datetime_features(x_train)
    x_test, _ = make_datetime_features(x_test)

    #  should put some more tests in here !!! TODO
    assert x_train.shape[1] == x_test.shape[1]

    """
    Now we can save our dataset

    We might like to save a bit more, such as unscaled features_path
    For now we just save the data our model will use to learn & predict
    """

    output = {'x_train' : x_train,
              'x_test' : x_test,
              'y_train' : y_train,
              'y_test' : y_test}

    for name, data in output.items():
        path = os.path.join('ff_data', name+'.csv')
        print('saving {}'.format(name))
        data.to_csv(path)
