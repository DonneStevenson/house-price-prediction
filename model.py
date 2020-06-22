from joblib import dump, load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


def train_model(X, y):
    params = {"loss": "ls",
              "learning_rate": 0.01,
              "n_estimators": 1500,
              "subsample": 0.8,
              "min_samples_split": 3,
              "max_depth": 10,
              "verbose": 0,
              "random_state": 1,
              "max_features": 0.6
              }

    gbr = GradientBoostingRegressor(**params)

    gbr_fit = gbr.fit(X, y.values.flatten())

    return gbr_fit


def predict_y(X, fitted_model):
    y_pred = fitted_model.predict(X)
    return y_pred


def validate_model(predictions, y):
    print(f"MAE of model:  {mean_absolute_error(y, predictions)}.")


def load_model_sklearn(path):
    fitted_model = load(open(path, 'rb'))
    return fitted_model


def save_model_sklearn(model, path):
    dump(model, open(path, 'wb'))
