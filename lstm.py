import numpy as np
import pandas as pd


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
    sequences = [raw_data.shift(-step) for step in range(timestep)]

    #  optionally reverse the order of the features
    #  this can improve performance
    #  see Sutskever (2016)
    if reverse:
        sequences = sequences[::-1]
    #  join sequences together
    sequences = pd.concat(sequences, axis=1).dropna()
    #  save the index
    index = sequences.index
    #  turn into a np array
    sequences = np.array(sequences).reshape(-1,
                                            timestep,
                                            feature_dim)

    #  we remove the last few features
    #  this is because we don't have targets for these
    #  the -1 is there because there is already a 1 step lag
    #  between features & target
    sequences = sequences[:-timestep+1]
    index = index[:-timestep+1]

    return sequences, index

def generate_target_sequences(raw_data, timestep):
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
    sequences = [raw_data.shift(step) for step in range(timestep)]
    #  reverse the sequences to get correct order
    sequences = sequences[::-1]
    #  join it all together
    sequences = pd.concat(sequences, axis=1).dropna()
    #  save the index
    index = sequences.index
    #  make a numpy array
    sequences = np.array(sequences).reshape(-1, timestep, 1)

    #  we remove the last few features
    #  this is because we don't have targets for these
    #  the -1 is there because there is already a 1 step lag
    #  between features & target
    sequences = sequences[timestep-1:]
    index = index[timestep-1:]
    return sequences, index

if __name__ == '__main__':
    test_seq = np.arange(100)
