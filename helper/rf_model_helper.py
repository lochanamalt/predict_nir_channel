"""
@author: Lochana Marasinghe
@date: 9/24/2025
@description: 
"""
from sklearn.ensemble import RandomForestRegressor


def get_rf_model(hyperparameters):
    rf_model = RandomForestRegressor(
        n_estimators=hyperparameters["n_estimators"],
        max_features=hyperparameters["max_features"],
        max_depth=hyperparameters["max_depth"],
        min_samples_split=hyperparameters["min_samples_split"],
        min_samples_leaf=hyperparameters["min_samples_leaf"],
        bootstrap=hyperparameters["bootstrap"],
        random_state=123,
        n_jobs=-1
    )
    return rf_model


def split_dataset(dataset):
    X = dataset.drop(columns=["uav_nir"])
    y = dataset["uav_nir"]
    return X, y
