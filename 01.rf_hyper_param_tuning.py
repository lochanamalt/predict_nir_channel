"""
@author: Lochana Marasinghe
@date: 9/24/2025
@description: 
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_excel("data/rf_data_with_features.xlsx")

# Split predictors and response
X = df.drop(columns=["uav_nir"])
y = df["uav_nir"]

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

param_grid = {
    "n_estimators": [200, 500, 1000, 1200],
    "max_features": [2, 3, 10, 17],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

rf = RandomForestRegressor(random_state=123)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                  # 5-fold CV
    scoring="r2",           # use R² as metric
    n_jobs=-1,              # parallel
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best CV R²:", grid_search.best_score_)

best_rf = grid_search.best_estimator_
# Best hyperparameters for whole dataset: {'bootstrap': True, 'max_depth': 10, 'max_features': 17, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1200}
# Best hyperparameters with hour : {'bootstrap': True, 'max_depth': 10, 'max_features': 17, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1000}
y_test_pred = best_rf.predict(X_test)

print("Test R²:", r2_score(y_test, y_test_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
