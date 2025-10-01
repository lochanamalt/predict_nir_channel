"""
@author: Lochana Marasinghe
@date: 9/24/2025
@description: 
"""


TOP_FEATURES_PYTHON = [
    "BB_true_diff",
    "RR_true_diff",
    "GG_true_diff",
    "RN_ratio",
    "GB_ratio",
    "BN_inter",
    "BN_ratio",
    "RB_inter",
    "GN_ratio",
    "RN_inter",
    "GB_inter",
    "R_noir2",
    "R_noir",
    "B_noir2",
    "R_noir_log"
]

TOP_FEATURES_PYTHON_WITH_HOUR = TOP_FEATURES_PYTHON + ["hour"]

TOP_FEATURES_R = [
    "BB_true_diff",
    "RR_true_diff",
    "GG_true_diff",
    "RN_ratio",
    "GB_ratio",
    "BN_ratio",
    "BN_inter",
    "RB_inter",
    "RN_inter",
    "GN_ratio",
    "R_noir2",
    "B_noir2",
    "B_noir",
    "B_noir_log",
    "R_noir",
    "R_noir_log"
]

TOP_FEATURES_R_WITH_HOUR = TOP_FEATURES_R + ["hour"]


all_features_with_hour_best_hyper_params = {'bootstrap': True, 'max_depth': 10, 'max_features': 17, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1000}


all_features_best_hyper_params = {'bootstrap': True, 'max_depth': 10, 'max_features': 17, 'min_samples_leaf': 1,
                                  'min_samples_split': 2,'n_estimators': 1200}
top_features_python_best_hyper_params = {'bootstrap': True, 'max_depth': 10, 'max_features': 3, 'min_samples_leaf': 1,
                                    'min_samples_split': 2, 'n_estimators': 200}
top_features_r_best_hyper_params = {'bootstrap': True, 'max_depth': None, 'max_features': 2, 'min_samples_leaf': 1,
                      'min_samples_split': 5, 'n_estimators': 200}
