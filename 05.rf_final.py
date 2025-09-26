"""
@author: Lochana Marasinghe
@date: 9/24/2025
@description: 
"""
import joblib
import pandas as pd

from constants.parameters import TOP_FEATURES_R, top_features_r_best_hyper_params
from helper.rf_model_helper import get_rf_model, split_dataset

if __name__ == "__main__":
    # Load the training data from R
    df = pd.read_excel("data/rf_data_with_features.xlsx")

    df_top_features_r = df[TOP_FEATURES_R + ["uav_nir"]]

    X, y = split_dataset(df_top_features_r)

    rf_final_full  = get_rf_model(top_features_r_best_hyper_params)

    rf_final_full.fit(X, y)  # use the whole dataset

    joblib.dump(rf_final_full, "saved_models/rf_final_full_predict_nir.pkl")









