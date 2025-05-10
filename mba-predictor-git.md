MBA Decision Data Challenge Spring 2025
================
Owen Hom
2025-02-10

Model \#1

Libraries

library(tidyverse)
library(caret)
library(xgboost)
library(themis)
library(recipes)

mba_data <- read.csv("mba_decision_dataset.csv")

engineer_features <- function(data) {
  data <- data %>%
    mutate(
      Age_Scaled = scale(Age),
      GPA_Scaled = scale(Undergraduate.GPA),
      Test_Score_Scaled = scale(GRE.GMAT.Score),
      Age_Group = cut(Age, breaks = c(0, 25, 30, 100), 
                      labels = c("Young", "Mid", "Senior"))
    )
  categorical_vars <- c("Gender", "Undergraduate.Major", "MBA.Funding.Source", "Age_Group")
  data[categorical_vars] <- lapply(data[categorical_vars], as.factor)
  return(data)
}

mba_data <- engineer_features(mba_data)
mba_data$Decided.to.Pursue.MBA <- factor(mba_data$Decided.to.Pursue.MBA, levels = c("No", "Yes"))

mba_features <- mba_data %>%
  select(-matches("Decided.to.Pursue.MBA"), -Person.ID)

set.seed(123)
trainIndex <- createDataPartition(mba_data$Decided.to.Pursue.MBA, p = 0.8, list = FALSE, times = 1)
trainData <- mba_features[trainIndex, ]
testData <- mba_features[-trainIndex, ]
trainData$Decided.to.Pursue.MBA <- mba_data$Decided.to.Pursue.MBA[trainIndex]
testData$Decided.to.Pursue.MBA <- mba_data$Decided.to.Pursue.MBA[-trainIndex]

recipe_obj <- recipe(Decided.to.Pursue.MBA ~ ., data = trainData) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(Decided.to.Pursue.MBA, over_ratio = 0.8)

prep_obj <- prep(recipe_obj)
trainData_balanced <- bake(prep_obj, new_data = NULL)

train_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "down",
  search = "grid"
)

xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 4, 5),
  eta = c(0.01, 0.05),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

xgb_model <- train(
  Decided.to.Pursue.MBA ~ .,
  data = trainData_balanced,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  metric = "ROC",
  verbose = FALSE
)

test_recipe <- recipe(Decided.to.Pursue.MBA ~ ., data = testData) %>%
  step_dummy(all_nominal_predictors())

test_prep <- prep(test_recipe)
testData_processed <- bake(test_prep, new_data = NULL)

predictions <- predict(xgb_model, testData_processed)
predictions_prob <- predict(xgb_model, testData_processed, type = "prob")

conf_matrix <- confusionMatrix(predictions, testData$Decided.to.Pursue.MBA)
print(conf_matrix)

library(pROC)
roc_obj <- roc(testData$Decided.to.Pursue.MBA, predictions_prob$Yes)
auc_value <- auc(roc_obj)
print(paste("AUC:", round(auc_value, 4)))

importance <- varImp(xgb_model)
print("Top 10 Most Important Features:")
print(importance$importance %>%
        as.data.frame() %>%
        rownames_to_column("Feature") %>%
        arrange(desc(Overall)) %>%
        head(10))

library(car)

mba_data <- read.csv("mba_decision_dataset.csv", 
                     colClasses = c("Person.ID" = "integer",
                                    "Age" = "integer",
                                    "Gender" = "character",
                                    "Undergraduate.Major" = "character",
                                    "Undergraduate.GPA" = "numeric",
                                    "GRE.GMAT.Score" = "integer",
                                    "MBA.Funding.Source" = "character",
                                    "Decided.to.Pursue.MBA." = "character"),
                     check.names = TRUE)

str(mba_data)

mba_data$Gender <- factor(mba_data$Gender)
mba_data$Undergraduate.Major <- factor(mba_data$Undergraduate.Major)
mba_data$MBA.Funding.Source <- factor(mba_data$MBA.Funding.Source)
mba_data$Decided.to.Pursue.MBA. <- factor(mba_data$Decided.to.Pursue.MBA.)

X <- model.matrix(~ Gender + Undergraduate.Major + MBA.Funding.Source + 
                   Age + Undergraduate.GPA + GRE.GMAT.Score - 1, data = mba_data)

y <- factor(mba_data$Decided.to.Pursue.MBA.)

model_data <- data.frame(X, MBA_Decision = y)

set.seed(123)
train_index <- createDataPartition(model_data$MBA_Decision, p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

model <- glm(MBA_Decision ~ ., data = train_data, family = "binomial")

print(summary(model))

predictions_prob <- predict(model, newdata = test_data, type = "response")
predictions_class <- factor(ifelse(predictions_prob > 0.5, "Yes", "No"), 
                            levels = levels(test_data$MBA_Decision))

conf_matrix <- confusionMatrix(predictions_class, test_data$MBA_Decision)
print(conf_matrix)

roc_obj <- roc(as.numeric(test_data$MBA_Decision == "Yes"), predictions_prob)
auc_value <- auc(roc_obj)
print(paste("AUC:", round(auc_value, 3)))

coef_summary <- summary(model)$coefficients
importance <- abs(coef_summary[, "z value"])
importance_df <- data.frame(
  Feature = rownames(coef_summary),
  Importance = importance
)
importance_df <- importance_df[order(-importance_df$Importance), ]
