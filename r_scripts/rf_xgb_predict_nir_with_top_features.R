

library(readxl)
combined_extracted_plot_vi_with_uav <- read_excel("combined_extracted_plot_vi_with_uav.xlsx")

combined_extracted_plot_vi_with_uav$R_noir2 <- combined_extracted_plot_vi_with_uav$R_noir^2
combined_extracted_plot_vi_with_uav$G_noir2 <- combined_extracted_plot_vi_with_uav$G_noir^2
combined_extracted_plot_vi_with_uav$B_noir2 <- combined_extracted_plot_vi_with_uav$B_noir^2
combined_extracted_plot_vi_with_uav$NIR_true2 <- combined_extracted_plot_vi_with_uav$NIR_true^2

combined_extracted_plot_vi_with_uav$R_noir_log <- log(combined_extracted_plot_vi_with_uav$R_noir + 1)
combined_extracted_plot_vi_with_uav$G_noir_log <- log(combined_extracted_plot_vi_with_uav$G_noir + 1)
combined_extracted_plot_vi_with_uav$B_noir_log <- log(combined_extracted_plot_vi_with_uav$B_noir + 1)
combined_extracted_plot_vi_with_uav$NIR_true_log <- log(combined_extracted_plot_vi_with_uav$NIR_true + 1)

# Simple ratios
combined_extracted_plot_vi_with_uav$RN_ratio <- combined_extracted_plot_vi_with_uav$R_noir / combined_extracted_plot_vi_with_uav$NIR_true
combined_extracted_plot_vi_with_uav$GN_ratio <- combined_extracted_plot_vi_with_uav$G_noir / combined_extracted_plot_vi_with_uav$NIR_true
combined_extracted_plot_vi_with_uav$BN_ratio <- combined_extracted_plot_vi_with_uav$B_noir / combined_extracted_plot_vi_with_uav$NIR_true
combined_extracted_plot_vi_with_uav$GB_ratio <- combined_extracted_plot_vi_with_uav$G_noir / combined_extracted_plot_vi_with_uav$B_noir

# Differences
combined_extracted_plot_vi_with_uav$GG_true_diff <- combined_extracted_plot_vi_with_uav$G_noir - combined_extracted_plot_vi_with_uav$G_true
combined_extracted_plot_vi_with_uav$BB_true_diff <- combined_extracted_plot_vi_with_uav$B_noir - combined_extracted_plot_vi_with_uav$B_true
combined_extracted_plot_vi_with_uav$RR_true_diff <- combined_extracted_plot_vi_with_uav$R_noir - combined_extracted_plot_vi_with_uav$R_true

#interactions
combined_extracted_plot_vi_with_uav$RG_inter <- combined_extracted_plot_vi_with_uav$R_noir * combined_extracted_plot_vi_with_uav$G_noir
combined_extracted_plot_vi_with_uav$RN_inter <- combined_extracted_plot_vi_with_uav$R_noir * combined_extracted_plot_vi_with_uav$NIR_true
combined_extracted_plot_vi_with_uav$RB_inter <- combined_extracted_plot_vi_with_uav$R_noir * combined_extracted_plot_vi_with_uav$B_noir
combined_extracted_plot_vi_with_uav$GB_inter <- combined_extracted_plot_vi_with_uav$G_noir * combined_extracted_plot_vi_with_uav$B_noir
combined_extracted_plot_vi_with_uav$GN_inter <- combined_extracted_plot_vi_with_uav$G_noir * combined_extracted_plot_vi_with_uav$NIR_true
combined_extracted_plot_vi_with_uav$BN_inter <- combined_extracted_plot_vi_with_uav$B_noir * combined_extracted_plot_vi_with_uav$NIR_true



# 25 features
features = c("G_noir", "B_noir", "R_noir", "NIR_true", "R_noir2", "G_noir2", "B_noir2",
             "NIR_true2", "R_noir_log", "G_noir_log", "B_noir_log", "NIR_true_log",
             "RN_ratio", "GN_ratio", "BN_ratio", "GB_ratio", "GG_true_diff", "BB_true_diff",
             "RR_true_diff", "RG_inter", "RN_inter", "RB_inter", "GB_inter", "GN_inter", "BN_inter" )

data_model <- combined_extracted_plot_vi_with_uav[, c("uav_nir", features)]

library(writexl)
write_xlsx(data_model, "rf_data_with_features.xlsx")

# ------------------------
# Train/test split
# ------------------------
set.seed(123)
n <- nrow(data_model)
train_idx <- sample(1:n, size = 0.7 * n)
train <- data_model[train_idx, ]
test  <- data_model[-train_idx, ]

# ------------------------
# Random Forest model
# ------------------------

library(randomForest)


# -------------------------
# Fine tuning RF model

#install.packages("caret")

library(caret)

#cross validation
ctrl <- trainControl(method = "cv", number = 5)

rf_cv <- train(
  uav_nir ~ .,
  data = data_model,
  method = "rf",
  trControl = ctrl,
  ntree = 1000,
  tuneLength = 10
)

print(rf_cv)

print(rf_cv$bestTune)
#mtry = 17

rf_imp <- varImp(rf_cv, scale = TRUE)
print(rf_imp)
plot(rf_imp, top = 10)

rf_imp_df <- as.data.frame(rf_imp$importance)  # caret varImp stores it in $importance

# Add row names as a column
rf_imp_df$Feature <- rownames(rf_imp_df)

# Reorder columns
rf_imp_df <- rf_imp_df[, c("Feature", colnames(rf_imp_df)[1:(ncol(rf_imp_df)-1)])]

# Save to CSV
write.csv(rf_imp_df, "rf_variable_importance.csv", row.names = FALSE)


# top 10 features
top_features = c( "BB_true_diff", "RR_true_diff",  "GG_true_diff", 
                  "RN_ratio","GB_ratio", "BN_ratio", "BN_inter","RB_inter", "RN_inter",
                  "GN_ratio")

top_features2 = c( "BB_true_diff", "RR_true_diff",  "GG_true_diff", 
                  "RN_ratio","GB_ratio", "BN_ratio", "BN_inter","RB_inter", "RN_inter",
                  "GN_ratio", "R_noir2", "B_noir2", "B_noir", "B_noir_log",
                  "R_noir", "R_noir_log")

set.seed(123)
rf_model <- randomForest(
  uav_nir ~ .,
  data = train,
  ntree = 1000,
  importance = TRUE,
  mtry = 17
)

plot(rf_model)
print(rf_model)



# Predict on test set
rf_preds <- predict(rf_model, newdata = test)
# Compute test-set R²
rf_r2 <- 1 - sum((test$uav_nir - rf_preds)^2) / sum((test$uav_nir - mean(test$uav_nir))^2)
cat("Random Forest test R²:", round(rf_r2, 4), "\n")
# Random Forest test R²: 0.1582

importance(rf_model)

# Variable importance plot
varImpPlot(rf_model)
# It is confirmed that  "BB_true_diff", "RR_true_diff", "R_noir",  "GG_true_diff" are the top 4 features


# ------------------------
# RF training with top 10 features
# ------------------------

top_feature_data_model <- combined_extracted_plot_vi_with_uav[, c("uav_nir", top_features)]
top_feature_data_model2 <- combined_extracted_plot_vi_with_uav[, c("uav_nir", top_features2)]

set.seed(123)
train_top_features <- top_feature_data_model[train_idx, ]
test_top_features  <- top_feature_data_model[-train_idx, ]

train_top_features2 <- top_feature_data_model2[train_idx, ]
test_top_features2  <- top_feature_data_model2[-train_idx, ]

# Fine tuning RF model with top features

rf_cv_top_features <- train(
  uav_nir ~ .,
  data = top_feature_data_model,
  method = "rf",
  trControl = ctrl,
  ntree = 1200,
  tuneLength = 9
)

rf_cv_top_features2 <- train(
  uav_nir ~ .,
  data = top_feature_data_model2,
  method = "rf",
  trControl = ctrl,
  ntree = 1200,
  tuneLength = 9
)


print(rf_cv_top_features) # best R_sq = 0.2446198
print(rf_cv_top_features$bestTune)
# mtry = 2

print(rf_cv_top_features2) # best R_sq = 0.2595208
print(rf_cv_top_features2$bestTune)
# mtry = 2

saveRDS(rf_cv_top_features2, file = "best_rf_caret_model_with_top_features.rds")

print(rf_cv) #best R_sq = 0.2354644
saveRDS(rf_cv, file = "best_rf_caret_model.rds")
# model with top_features2 performs better than other models


set.seed(123)
rf_model_top_features <- randomForest(
  uav_nir ~ .,
  data = train_top_features2,
  ntree = 1000,
  importance = TRUE,
  mtry = 2
)

plot(rf_model_top_features)
print(rf_model_top_features)

# Predict on test set
rf_preds_top_features2 <- predict(rf_model_top_features, newdata = test_top_features2)

# Compute test-set R²
rf_r2_top_features2 <- 1 - sum((test_top_features2$uav_nir - rf_preds_top_features2)^2) / sum((test_top_features2$uav_nir - mean(test_top_features2$uav_nir))^2)
cat("Random Forest test with top features R²:", round(rf_r2_top_features2, 4), "\n")
#Random Forest test with top features R²: 0.2004

# ---- Final production model -------
# Use the whole dataset

write_xlsx(top_feature_data_model2, "rf_data_with_top_features.xlsx")

print(rf_cv_top_features2) #best R_sq = 0.2595208
# Following are the performance of the model
# Random Forest 
# 
# 738 samples
# 16 predictor
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 590, 590, 591, 590, 591 
# Resampling results across tuning parameters:
#   
#   mtry  RMSE        Rsquared   MAE       
# 2    0.05056626  0.2595208  0.03801913
# 3    0.05058796  0.2597232  0.03801886
# 5    0.05091181  0.2527579  0.03814992
# 7    0.05094874  0.2518402  0.03822862
# 9    0.05110030  0.2484470  0.03829325
# 10    0.05136530  0.2424796  0.03851546
# 12    0.05149708  0.2393968  0.03851432
# 14    0.05156718  0.2381928  0.03857559
# 16    0.05172748  0.2357228  0.03871915
# 
# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 2.

set.seed(123)
rf_final <- randomForest(
  uav_nir ~ .,
  data = top_feature_data_model2,
  ntree = 1000,
  mtry = 2,      # best from CV
  importance = TRUE
)


print(rf_final)
saveRDS(rf_final, "rf_final_topfeatures.rds")

# ------------------------
#  XGBoost model
# ------------------------

library(xgboost)

train_matrix <- as.matrix(train[, top_features2])
train_label  <- train$uav_nir
test_matrix  <- as.matrix(test[, top_features2])
test_label   <- test$uav_nir

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)

xgb_model <- xgboost(
  data = dtrain,
  objective = "reg:squarederror",
  nrounds = 100,
  max_depth =5,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.9,
  verbose = 0
)


# Predictions
xgb_preds <- predict(xgb_model, test_matrix)

# Compute test-set R²
xgb_r2 <- 1 - sum((test_label - xgb_preds)^2) / sum((test_label - mean(test_label))^2)
cat("XGBoost test R²:", round(xgb_r2, 4), "\n")

#XGBoost test R²: 0.1554 , RF is better

#-----------------------------------------
# Cross validation and fine tuning for XGB

ctrl_xgb <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

# Define XGBoost grid
xgb_grid <- expand.grid(
  nrounds = c(100, 200),      # boosting iterations
  max_depth = c(3, 4, 5),     # tree depth
  eta = c(0.05, 0.1),         # learning rate
  gamma = 0,                  # minimum loss reduction
  colsample_bytree = c(0.7, 0.9), # column sampling
  min_child_weight = 1,
  subsample = 0.8
)

xgb_cv <- train(
  uav_nir ~ .,
  data = data_model,
  method = "xgbTree",
  trControl = ctrl_xgb,
  tuneGrid = xgb_grid,
  metric = "RMSE"
)


print(xgb_cv)
plot(xgb_cv)  # visualize tuning results

best_params <- xgb_cv$bestTune
cat("Best parameters:\n")
print(best_params)

print(xgb_cv$results)
print(xgb_cv$bestTune)

xgb_imp <- varImp(xgb_cv, scale = TRUE)
print(xgb_imp)
plot(xgb_imp, top = 10)

