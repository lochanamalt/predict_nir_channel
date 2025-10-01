"""
@author: Lochana Marasinghe
@date: 9/24/2025
@description: 
"""
import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.model_selection import cross_val_score

from constants.parameters import TOP_FEATURES_PYTHON, TOP_FEATURES_R, all_features_best_hyper_params, \
    top_features_python_best_hyper_params, top_features_r_best_hyper_params, all_features_with_hour_best_hyper_params, \
    TOP_FEATURES_PYTHON_WITH_HOUR, TOP_FEATURES_R_WITH_HOUR
from helper.rf_model_helper import get_rf_model, split_dataset


def random_forest_regressor(dataset, hyperparameters):
    # Split predictors and response
    X, y = split_dataset(dataset)

    rf_model = get_rf_model(hyperparameters)

    # Compute cross-validated R²
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=123)
    scores = cross_val_score(rf_model, X, y, cv=cv, scoring="r2")
    print("Mean R²:", scores.mean(), "Std:", scores.std())

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    rf_model.fit(X_train, y_train)
    y_train_pred = rf_model.predict(X_train)
    print("Train R²:", round(r2_score(y_train, y_train_pred), 4))
    print("Train MSE:", round(mean_squared_error(y_train, y_train_pred), 6))

    # Test performance
    y_test_pred = rf_model.predict(X_test)
    print("Test R²:", round(r2_score(y_test, y_test_pred), 4))
    print("Test MSE:", round(mean_squared_error(y_test, y_test_pred), 6))


if __name__ == "__main__":
    # Load the training data from R
    df = pd.read_excel("data/rf_data_with_features.xlsx")

    random_forest_regressor(df, all_features_with_hour_best_hyper_params)
    # Mean R²: 0.21583950820061268
    # Std: 0.0697806743145798
    # Train R²: 0.8228
    # Train MSE: 0.000604
    # Test R²: 0.2687
    # Test MSE: 0.002472

    df_top_features_python = df[TOP_FEATURES_PYTHON_WITH_HOUR + ["uav_nir"]]
    random_forest_regressor(df_top_features_python, top_features_python_best_hyper_params)
    # Mean R²: 0.21971810628463112
    # Std: 0.06884349502957038
    # Train R²: 0.8013
    # Train MSE: 0.000677
    # Test R²: 0.2802
    # Test MSE: 0.002433

    df_top_features_r = df[TOP_FEATURES_R_WITH_HOUR + ["uav_nir"]]
    random_forest_regressor(df_top_features_r, top_features_r_best_hyper_params)
    # Mean R²: 0.23630120783812006
    # Std: 0.07425206930739775
    # Train R²: 0.8185
    # Train MSE: 0.000619
    # Test R²: 0.3285
    # Test MSE: 0.00227

    # This model is the best rf model








