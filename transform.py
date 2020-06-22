import pandas as pd


def transform_train_data(data):
    threshold = (165 + 1.5 * (165 - 99))*1000
    train_data = data[["neighbourhood", "size", "bedrooms", "bathrooms", "price"]].copy(deep=True)

    train_data = train_data.loc[train_data["price"] <= threshold]

    train_data = train_data.loc[(train_data["size"] > 0) & (train_data["size"] <= 400_000)]

    train_data = train_data.loc[ ((train_data["bathrooms"] <= (train_data["bedrooms"]*2))) | ((train_data["bathrooms"] ==1) & (train_data["bedrooms"]==0 ))]
    train_data = train_data.loc[(train_data["bathrooms"] > 0) | (train_data["bedrooms"] > 0)]

    train_data["bathratio"] = train_data["bathrooms"] / (train_data["bathrooms"] + train_data["bedrooms"])
    train_data["bathratio"] = train_data["bathratio"].fillna(0)

    possible_categories = ('SNR', 'ZMS', 'PLY')
    train_data["neighbourhood"] = train_data["neighbourhood"].astype(pd.CategoricalDtype(categories=possible_categories))
    train_data = pd.get_dummies(train_data, columns=["neighbourhood"])

    return train_data


def transform_test_features(data):
    test_data = data[["neighbourhood", "size", "bedrooms", "bathrooms"]].copy(deep=True)
    test_data["bathratio"] = test_data["bathrooms"] / (test_data["bathrooms"] + test_data["bedrooms"])
    test_data["bathratio"] = test_data["bathratio"].fillna(0)

    possible_categories = ('SNR', 'ZMS', 'PLY')
    test_data["neighbourhood"] = test_data["neighbourhood"].astype(pd.CategoricalDtype(categories=possible_categories))
    test_data = pd.get_dummies(test_data, columns=["neighbourhood"])

    return test_data
