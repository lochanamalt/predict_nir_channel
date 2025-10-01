"""
@author: Lochana Marasinghe
@date: 9/24/2025
@description: 
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from constants.parameters import TOP_FEATURES_PYTHON, TOP_FEATURES_R, \
    TOP_FEATURES_R_WITH_HOUR, TOP_FEATURES_PYTHON_WITH_HOUR

param_grid = {
    "n_estimators": [200, 500, 1000, 1200],
    "max_features": [2, 3, 10, 17],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

def run_grid_search(dataset):

    X = dataset.drop(columns=["uav_nir"])
    y = dataset["uav_nir"]

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    rf = RandomForestRegressor(random_state=123)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,  # 5-fold CV
        scoring="r2",  # use R² as metric
        n_jobs=-1,  # parallel
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("Best hyperparameters:", grid_search.best_params_)
    print("Best CV R²:", grid_search.best_score_)

    best_rf = grid_search.best_estimator_
    # Best hyperparameters for whole dataset: {'bootstrap': True, 'max_depth': 10, 'max_features': 17, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1200}

    y_test_pred = best_rf.predict(X_test)

    print("Test R²:", r2_score(y_test, y_test_pred))
    print("Test MSE:", mean_squared_error(y_test, y_test_pred))

if __name__ == "__main__":
    df = pd.read_excel("data/rf_data_with_features.xlsx")

    df_top_python = df[TOP_FEATURES_PYTHON_WITH_HOUR + ["uav_nir"]]
    df_top_r = df[TOP_FEATURES_R_WITH_HOUR + ["uav_nir"]]

    run_grid_search(df_top_python)
    # Best hyperparameters: {'bootstrap': True, 'max_depth': None, 'max_features': 3, 'min_samples_leaf': 1,
    #                   'min_samples_split': 2, 'n_estimators': 200}
    # Best CV R²: 0.19361396255012361
    # Test R²: 0.2813646882075539
    # Test MSE: 0.0024294216502943606

    #run_grid_search(df_top_r)

    # Best hyperparameters: {'bootstrap': True, 'max_depth': None, 'max_features': 3, 'min_samples_leaf': 1,
    # 'min_samples_split': 2, 'n_estimators': 200}
    # Best CV R²: 0.19081073523827483
    # Test R²: 0.32247694272595573
    # Test MSE: 0.0022904373844498454


