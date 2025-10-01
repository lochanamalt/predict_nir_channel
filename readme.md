
# Estimating NIR Channel Based on Raspberry Pi NoIR Camera Parameters

## Overview

This project estimates the **Near-Infrared (NIR) channel** using features derived from the **Raspberry Pi NoIR camera**.
The dependent variable is the **UAV_NIR channel**, and the model is trained using a combination of **raw and derived features**.

---

## Predictors (25 Features)

The following features were engineered and used as predictors:

1. R_noir
2. G_noir
3. B_noir
4. NIR_true*
5. (R_noir)²
6. (G_noir)²
7. (B_noir)²
8. (B_true)²
9. log(R_noir + 1)
10. log(G_noir + 1)
11. log(B_noir + 1)
12. log(NIR_true + 1)
13. R_noir / NIR_true*
14. G_noir / NIR_true*
15. B_noir / NIR_true*
16. G_noir / B_noir
17. R_noir – R_true*
18. G_noir – G_true*
19. B_noir – B_true*
20. R_noir * G_noir
21. R_noir * B_noir
22. R_noir * NIR_true*
23. G_noir * B_noir
24. G_noir * NIR_true*
25. B_noir * NIR_true*

*calculated using coefficients from the referenced paper (https://doi.org/10.1080/15538362.2018.1502720), method 2


---

## Hyperparameter Tuning and Feature Importance

### Grid Search Results

Tuning was performed using Grid Search with the full set of features.

* **Best hyperparameters (Python, full dataset):**

```python
{'bootstrap': True, 
 'max_depth': 10, 
 'max_features': 17, 
 'min_samples_leaf': 1, 
 'min_samples_split': 2, 
 'n_estimators': 1200}
```
### Feature Importance

* Training/Testing split: **80% train, 20% test**
* Feature importance was performed in Python (scikit-learn) using the best hyperparameters obtained.
* Moreover, hyperparameters tuned and performed feature importance in R (caret library).
* Now, we have 2 sets of best featured obtained from Python and R.
---

## Performance

### Python-based Best Features with Best Hyperparameters:

* Mean CV R²: 0.2197 (±0.0751)
* Train R²: 0.8013
* Train MSE: 0.000677
* Test R²: 0.2802
* Test MSE: 0.002433

### R-based Best Features with Best Hyperparameters

* Mean CV R²: 0.2363 (±0.0763)
* Train R²: 0.8185
* Train MSE: 0.000619
* Test R²: 0.3285
* Test MSE: 0.00227

✅ The **R-derived feature set** provided better performance.

### Selected Best Features

1. BB_true_diff
2. RR_true_diff
3. GG_true_diff
4. RN_ratio
5. GB_ratio
6. BN_ratio
7. BN_inter
8. RB_inter
9. RN_inter
10. GN_ratio
11. R_noir2
12. B_noir2
13. B_noir
14. B_noir_log
15. R_noir
16. R_noir_log
17. hour

---
## Final Random Forest Model

* Trained on the **whole dataset**
* Used the **best features (from R analysis)**
---
