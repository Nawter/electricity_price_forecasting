import numpy as np
import pandas as pd
import sklearn.preprocessing

from keras_models import make_lstm


"""
I use two helper functions to turn the feature pd.DataFrame and target pd.Series
into 3D arrays for use in an LSTM

The arrays are of the shape
features = (num_samples, timestep, num_features)
target = (num_samples, timestep, 1)

Could probably use a single function but for simplicity and understanding
I use two!
"""

def generate_feature_sequences(raw_data, timestep, reverse=True):
    """
    args
        raw_data : pd.DataFrame
        timestep : int
        reverse  : Boolean

    returns
        sequences : np array
        index     : pandas index object
    """
    #  get the number of features
    feature_dim = raw_data.shape[1]

    #  lag the features backwards
    sequences = [raw_data.shift(step) for step in range(timestep)]

    #  optionally reverse the order of the features
    #  this can improve performance
    #  see Sutskever (2016)
    if reverse:
        sequences = sequences[::-1]
    #  join sequences together
    sequences = pd.concat(sequences, axis=1)
    #  save the index
    index = sequences.index
    #  turn into a np array
    sequences = np.array(sequences[timestep:-timestep]).reshape(-1,
                                            timestep,
                                            feature_dim)

    return sequences, index

def generate_target_sequences(raw_data, timestep, reverse=True):
    """
    args
        raw_data : pd.DataFrame
        timestep : int

    returns
        sequences : np array
        index     : pandas index object
    """

    target_dim = raw_data.shape[1]
    sequences = [raw_data.shift(-step) for step in range(timestep)]
    if reverse:
        #  reverse the sequences to get correct order
        sequences = sequences[::-1]
    #  join it all together
    sequences = pd.concat(sequences, axis=1)
    #  save the index
    index = sequences.index
    #  make a numpy array
    sequences = np.array(sequences[timestep:-timestep]).reshape(-1, timestep, target_dim)
    return sequences, index

def scale_data(raw_data):
    """
    args
        data (pd.Series or DataFrame)

    returns
        scaled_data (pd.DataFrame)
        scalers  (list)
    """

    scalers, scaled_data = [], []
    for name, series in features.iteritems():
        sclr = sklearn.preprocessing.StandardScaler()
        data = np.array(series).reshape(-1, 1)
        data = sclr.fit_transform(data)
        scaled_data.append(pd.DataFrame(data))
        scalers.append(sclr)

    scaled_data = pd.concat(scaled_data, axis=1)

    return scaled_data, scalers


def descale(scaled_data, scalers):
    """
    args
        data (pd.DataFrame)
        scalers (list)

    returns
        unscaled_data (pd.DataFrame)
    """
    unscaled_data = []
    for i, sclr in enumerate(scalers):
        data = np.array(scaled_data.iloc[:,i])
        unscaled = sclr.inverse_transform(data)
    unscaled_data = pd.concat(unscaled_data, axis=1)
    return unscaled_data


def test_sequence_gen(timestep, target, features):
    """
    Tests the two sequence generation function
    """
    target, t_index = generate_target_sequences(target, TIMESTEP)
    features, f_index = generate_feature_sequences(features, TIMESTEP, reverse=False)

    for f, t in zip(features, target):
        print('target {}'.format(t.flatten()))
        print('features {}'.format(f.flatten()))
        print('diff in target & feature {}'.format((t-f).flatten()))

    assert np.all(np.isnan(features)) == False

if __name__ == '__main__':
    TIMESTEP = 4

    features = pd.DataFrame(np.arange(0, 50), dtype=float)
    target = pd.Series(np.arange(10, 60), dtype=float)
    print('features shape is {}'.format(features.shape))
    print('target shape is {}'.format(target.shape))
    assert features.shape[0] == target.shape[0]

    features, f_scalers = scale_data(features)
    target, t_scalers = scale_data(target)

    features, f_index = generate_feature_sequences(features, TIMESTEP)
    target, t_index = generate_target_sequences(target, TIMESTEP)

    print('features shape is {}'.format(features.shape))
    print('target shape is {}'.format(target.shape))

    model = make_lstm(timestep=TIMESTEP,
                      input_length=features.shape[2],
                      layer_nodes=[10, 8, 6, 4],
                      dropout=0.0)

    model.fit(x=features, y=target, epochs=100, batch_size=10)

    prediction = pd.DataFrame(model.predict(x=features).reshape(-1, TIMESTEP))

    prediction = descale(prediction, t_scalers)
    target = descale(target.reshape(-1, TIMESTEP), t_scalers)

    for f, t, p in zip(features, target, prediction):
        print('features {}'.format(f.flatten()))
        print('target {}'.format(t.flatten()))
        print('prediction {}'.format(p.flatten()))
        print('diff in target & feature {}'.format((t-f).flatten()))
        print('diff in target & prediction {}'.format((t-p).flatten()))
