from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


models = {'random_forest': RandomForestRegressor,
          'linear': LinearRegression}


def get_model(model_id, **kwargs):
    model = models[model_id]
    return model(**kwargs)
