"""
@author: Lochana Marasinghe
@date: 9/24/2025
@description: 
"""
import pandas as pd
from matplotlib import pyplot as plt

from constants.parameters import all_features_best_hyper_params
from helper.rf_model_helper import split_dataset, get_rf_model

# Load the training data from R
df = pd.read_excel("data/rf_data_with_features.xlsx")

X, y = split_dataset(df)

rf_model = get_rf_model(all_features_best_hyper_params)

rf_model.fit(X, y)

# Extract importance
importance = rf_model.feature_importances_
features = X.columns

# Sort features by importance
sorted_idx = importance.argsort()[::-1]

print("Feature Importances (descending):")
for i in sorted_idx:
    print(f"{features[i]}: {importance[i]:.4f}")

"""
BB_true_diff: 0.1249
RR_true_diff: 0.1081
GG_true_diff: 0.0682
RN_ratio: 0.0673
GB_ratio: 0.0654
BN_inter: 0.0513
BN_ratio: 0.0494
RB_inter: 0.0395
GN_ratio: 0.0383
RN_inter: 0.0380
GB_inter: 0.0325
R_noir2: 0.0322
R_noir: 0.0320
B_noir: 0.0314
R_noir_log: 0.0312
B_noir_log: 0.0298
B_noir2: 0.0287
RG_inter: 0.0244
GN_inter: 0.0185
G_noir_log: 0.0167
G_noir2: 0.0164
G_noir: 0.0163
NIR_true_log: 0.0133
NIR_true: 0.0132
NIR_true2: 0.0129

"""

# Plot
plt.figure(figsize=(8,6))
plt.barh([features[i] for i in sorted_idx], importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()
