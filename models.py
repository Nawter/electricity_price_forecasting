from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression



def get_model(model_id, **kwargs):
    model = models[model_id]
    return model(**kwargs)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def make_feedforward(learning_rate=0.0001):
    dropout = 0.3

    input_shape = (8,)
    output_shape = (2,)

    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))

    print(model.summary())

    model.compile(loss='rmse',
                  optimizer=Adam(learning_rate),
                  metrics=['absolute_error'])


models = {'random_forest': RandomForestRegressor,
          'linear': LinearRegression}

