from model import *
from transform import *


def train(data_path="train.csv", model_path="model.joblib"):
    data = pd.read_csv(data_path)
    transform_data = transform_train_data(data)

    trained_model = train_model(transform_data.drop("price", axis=1), transform_data[["price"]])
    save_model_sklearn(trained_model, model_path)


if __name__ == "__main__":
    train()
