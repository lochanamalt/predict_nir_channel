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
    top_features_python_best_hyper_params, top_features_r_best_hyper_params
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

    random_forest_regressor(df, all_features_best_hyper_params)
    # Mean R²: 0.2042239876355485
    # Std: 0.07457178407170197
    # Train R²: 0.8166
    # Train MSE: 0.000625
    # Test R²: 0.2584
    # Test MSE: 0.002507

    df_top_features_python = df[TOP_FEATURES_PYTHON + ["uav_nir"]]
    random_forest_regressor(df_top_features_python, top_features_python_best_hyper_params)
    # Mean R²: 0.21065653775766613
    # Std: 0.075087135891057
    # Train R²: 0.7953
    # Train MSE: 0.000698
    # Test R²: 0.2934
    # Test MSE: 0.002389

    df_top_features_r = df[TOP_FEATURES_R + ["uav_nir"]]
    random_forest_regressor(df_top_features_r, top_features_r_best_hyper_params)
    # Mean R²: 0.21887255300018932
    # Std: 0.07634710707098032
    # Train R²: 0.815
    # Train MSE: 0.000631
    # Test R²: 0.3131
    # Test MSE: 0.002322

    # This model is the best rf model








