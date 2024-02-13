# Load necessary libraries
library(ISLR)
library(glmnet)
library(caret)

# Load the College dataset
data("College")
str (College)
summary (College)
View(College)

set.seed(123)
train_dataset <- createDataPartition(College$Grad.Rate, p = 0.7, list = FALSE)
train_dataset <- College[train_dataset, ]
test_indices <- setdiff(1:nrow(College), train_dataset)
test_dataset <- College[test_indices, ]


# Ridge Regression

# Response variables = Grad.Rate and other variables are predictor 
x_traindata <- model.matrix(Grad.Rate ~ ., data = train_dataset)[, -1]
y_traindata <- train_dataset$Grad.Rate

#cross-validation to find lambda values
ridge_cv <- cv.glmnet(x_traindata, y_traindata, alpha = 0)
lambda_min <- ridge_cv$lambda.min
lambda_min
lambda_1se <- ridge_cv$lambda.1se
lambda_1se

# Plot the results from cv.glmnet
plot(ridge_cv)

# Fit Ridge regression model
ridge_model <- glmnet(x_traindata, y_traindata, alpha = 0)
coef(ridge_model)

# RMSE for training set
ridge_train_pred <- predict(ridge_model, s = lambda_min, newx = x_traindata)
ridge_train_pred
ridge_train_rmse <- sqrt(mean((y_traindata - ridge_train_pred)^2))
ridge_train_rmse


# Calculate RMSE for test set
x_test <- model.matrix(Grad.Rate ~ ., data = test_data)[, -1]
y_test <- test_data$Grad.Rate
ridge_test_pred <- predict(ridge_model, s = lambda_min, newx = x_test)
ridge_test_rmse <- sqrt(mean((y_test - ridge_test_pred)^2))
ridge_test_rmse

# LASSO

# Use cross-validation to find optimal lambda values
lasso_cv <- cv.glmnet(x_traindata, y_traindata, alpha = 1)
lambda_min_lasso <- lasso_cv$lambda.min
lambda_1se_lasso <- lasso_cv$lambda.1se

# Plot the results from cv.glmnet
plot(lasso_cv)

# Fit LASSO regression model
lasso_model <- glmnet(x_traindata, y_traindata, alpha = 1)
coef(lasso_model)

# Calculate RMSE for training set
lasso_train_pred <- predict(lasso_model, s = lambda_min_lasso, newx = x_traindata)
lasso_train_rmse <- sqrt(mean((y_traindata - lasso_train_pred)^2))
lasso_train_rmse

# Calculate RMSE for test set
lasso_test_pred <- predict(lasso_model, s = lambda_min_lasso, newx = x_test)
lasso_test_rmse <- sqrt(mean((y_test - lasso_test_pred)^2))
lasso_test_rmse

# Comparison

# Print RMSE for both models
cat("Ridge Train RMSE:", ridge_train_rmse, "\n")
cat("Ridge Test RMSE:", ridge_test_rmse, "\n")
cat("LASSO Train RMSE:", lasso_train_rmse, "\n")
cat("LASSO Test RMSE:", lasso_test_rmse, "\n")

# Perform stepwise selection
stepwise_model <- step(lm(Grad.Rate ~ ., data = College), direction = "both")

# Print the summary of the selected model
summary(stepwise_model)
