import numpy as np
import pandas as pd

from keras_models import make_lstm


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
        raw_data : pd.Series
        timestep : int

    returns
        sequences : np array
        index     : pandas index object
    """

    #  check that we have been fed a pd.Series
    assert len(raw_data.shape) == 1
    #  shift all the data using pandas
    sequences = [raw_data.shift(-step) for step in range(timestep)]
    if reverse:
        #  reverse the sequences to get correct order
        sequences = sequences[::-1]
    #  join it all together
    sequences = pd.concat(sequences, axis=1)
    #  save the index
    index = sequences.index
    #  make a numpy array
    sequences = np.array(sequences[timestep:-timestep]).reshape(-1, timestep, 1)
    return sequences, index

def test_sequence_gen(timestep, target, features):
    """
    Tests the two sequence generation function
    Uses hardcoded TIMESTEP, features & targets
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

    features = pd.DataFrame(np.arange(0, 50))
    target = pd.Series(np.arange(10, 60))
    print('features shape is {}'.format(features.shape))
    print('target shape is {}'.format(target.shape))
    assert features.shape[0] == target.shape[0]

    #  do some super rough scaling
    features = (features - features.mean()) / features.std()
    target = (target - target.mean()) / target.std()

    features, f_index = generate_feature_sequences(features, TIMESTEP)
    target, t_index = generate_target_sequences(target, TIMESTEP)

    print('features shape is {}'.format(features.shape))
    print('target shape is {}'.format(target.shape))

    model = make_lstm(timestep=TIMESTEP,
                      input_length=features.shape[2],
                      layer_nodes=[10, 10, 10],
                      dropout=0.0)

    model.fit(x=features, y=target, epochs=100, batch_size=1)

    prediction =model.predict(x=features)

    for f, t, p in zip(features, target, prediction):
        print('features {}'.format(f.flatten()))
        print('target {}'.format(t.flatten()))
        print('prediction {}'.format(p.flatten()))
        print('diff in target & feature {}'.format((t-f).flatten()))
        print('diff in target & prediction {}'.format((t-p).flatten()))
