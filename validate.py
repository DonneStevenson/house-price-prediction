from model import *
from transform import *


def validate(data_path="test.csv", model_path="model.joblib"):
    data = pd.read_csv(data_path)
    valid_data = transform_test_features(data)
    fitted_model = load_model_sklearn(model_path)

    y_pred = predict_y(valid_data, fitted_model)
    validate_model(y_pred, data[["price"]])


if __name__ == "__main__":
    validate()

