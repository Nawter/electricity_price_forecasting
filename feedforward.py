import keras_models
from keras_models import make_ff_nn

if __name__ == 'main':

    model = make_ff_nn(input_length=x_train.shape[1],
                       layer_nodes=[100, 100],
                       output_neurons=(y_train.shape[1],),
                       dropout=0.35,
                       optimizer='Adam',
                       loss='mse')
