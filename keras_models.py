"""
Model uses relu as the activation functions for all layers except the last.
Relu is the current standard activation function used in modern neural networks.

The last layer has a linear activation function.  This is because we are doing
a regression problem, and we want our output to be able to range from -inf to
+inf.  If we used a relu on the output layer we wouldn't be able to predict
negative values.

Batch Normalization is used after each layer (before the activation function).
This is to scale the data input into the activation functions.
By scaling the batch we can help to keep the backpropagation gradients under control.

Adam is used as the optimizer - this is currently a standard optimizer used
in modern neural networks.  I have left the learning rate & other optimizer
hyperparameters at their default values.

"""
#  import stuff we need to make feedforward & LSTM mdoels
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential

#  import the tensorflow backend so we can prefer GPU training
import keras.backend.tensorflow_backend as KTF

def make_ff(input_length,
            layer_nodes,
            output_neurons,
            dropout=0.35,
            optimizer='Adam',
            loss='mse'):
    """
    Creates a feedforward neural network Keras model

    args
        input_length (int)  : used to define input shape
        layer_nodes (list) : number of nodes in each of the layers input & hidden
        output_neurons (int) : the number of predictions we want to make
        dropout (float) : the dropout rate to for the layer to layer connections
        optimizer (str) : reference to the Keras optimizer we want to use
        loss (str) : reference to the Keras loss function we want to use

    returns
        model (object) : the Keras feedforward neural network model
    """

    """
    First we make some changes to the TensorFLow backend config to try to
    get GPU training working
    """
    tf_config = KTF.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #  pass this tf config object into a new tf session object
    sess = tf.Session(config=tf_config)
    #  set the session as our new tf session
    KTF.set_session(sess)
    #  try to get TensorFlow to train on GPU
    with KTF.tf.device('/gpu:0'):
        """
        Now we can create our Keras model
        """
        #  create model object
        model = keras.models.Sequential()
        #  batch normalize the input data
        model.add(BatchNormalization())
        #  create the input layer
        model.add(Dense(layer_nodes[0], input_shape=(input_length,)))
        #  batch normalize the output of the input layer
        model.add(BatchNormalization())
        #  feed our batch normalized data into the activation function
        model.add(Activation('relu'))
        #  dropout some of the connections between the input layer and the
        #  first hidden layer
        model.add(Dropout(dropout))

        #  now we setup our hidden layers
        for nodes in hidden_layers[1:]:
            #  add the hidden layer
            model.add(Dense(nodes))
            #  batch norm the output of our hidden layer
            #  feed into our activation function
            #  dropout some connections to the next layer
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(dropout))

        #  add the output layer
        model.add(Dense(output_neurons))
        #  add our linear activation function so that we can predict negatives
        model.add(Activation('linear'))

        #  compile the model
        model.compile(optimizer=optimizer, loss=loss)
        a=1
        print(model.summary())

    return model


def make_lstm(timestep,
              input_length,
              layer_nodes,
              dropout=0.35,
              optimizer='Adam',
              loss='mse'):
    """
    Creates a Long Short Term Memory (LSTM) neural network Keras model

    args
        timestep (int) : the length of the sequence
        input_length (int) : used to define input shape
        layer_nodes (list) : number of nodes in each of the layers input & hidden
        dropout (float) : the dropout rate to for the layer to layer connections
        optimizer (str) : reference to the Keras optimizer we want to use
        loss (str) : reference to the Keras loss function we want to use

    returns
        model (object) : the Keras LSTM neural network model
    """

    model = Sequential()

    #  first we add the input layer
    model.add(LSTM(units=layer_nodes[0],
                   input_shape=(timestep, input_length),
                   return_sequences=True))
    #  batch norm to normalize data going into the actvation functions
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #  dropout some connections into the first hidden layer
    model.add(Dropout(dropout))

    #  now add hideen layers using the same strucutre
    for nodes in layer_nodes[1:]:
        model.add(LSTM(units=nodes, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

    #  add the output layer with a linear activation function
    #  we use a node size of 1 hard coded because we make one prediction
    #  per time step
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation('linear'))

    #  compile model using user defined loss function and optimizer
    model.compile(loss=loss, optimizer=optimizer)
    print(model.summary())

    return model
