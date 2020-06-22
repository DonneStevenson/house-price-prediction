from model import *
from transform import *


def predict(data_path="test.csv", model_path="model.joblib"):
    data = pd.read_csv(data_path)
    test_data = transform_test_features(data)
    fitted_model = load_model_sklearn(model_path)
    predictions = predict_y(test_data, fitted_model)
    pred_df = pd.DataFrame(predictions, columns=["price_predictions"])
    pred_df.to_csv("predictions.csv")


if __name__ == "__main__":
    predict()

