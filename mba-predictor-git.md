MBA Decision Data Challenge Spring 2025
================
Owen Hom
2025-02-10

Model \#1

Libraries

``` r
library(tidyverse)
```

    ## Warning: package 'ggplot2' was built under R version 4.4.2

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   3.5.1     ✔ tibble    3.2.1
    ## ✔ lubridate 1.9.3     ✔ tidyr     1.3.1
    ## ✔ purrr     1.0.2     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 4.4.2

    ## Loading required package: lattice
    ## 
    ## Attaching package: 'caret'
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(xgboost)
```

    ## Warning: package 'xgboost' was built under R version 4.4.2

    ## 
    ## Attaching package: 'xgboost'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     slice

``` r
library(themis)
```

    ## Warning: package 'themis' was built under R version 4.4.2

    ## Loading required package: recipes
    ## 
    ## Attaching package: 'recipes'
    ## 
    ## The following object is masked from 'package:stringr':
    ## 
    ##     fixed
    ## 
    ## The following object is masked from 'package:stats':
    ## 
    ##     step

``` r
library(recipes)
```

Load Data

``` r
mba_data <- read.csv("mba_decision_dataset.csv")
```

2/11 Sonnet 3.5

``` r
# Simplified feature engineering function - reducing complexity
engineer_features <- function(data) {
  # Apply transformations
  data <- data %>%
    mutate(
      # Standardize numerical variables
      Age_Scaled = scale(Age),
      GPA_Scaled = scale(Undergraduate.GPA),
      Test_Score_Scaled = scale(GRE.GMAT.Score),
      
      # Simplified age groups
      Age_Group = cut(Age, breaks = c(0, 25, 30, 100), 
                    labels = c("Young", "Mid", "Senior"))
    )
  
  # Convert categorical variables to factors
  categorical_vars <- c("Gender", "Undergraduate.Major", "MBA.Funding.Source", "Age_Group")
  data[categorical_vars] <- lapply(data[categorical_vars], as.factor)
  
  return(data)
}

# Process full dataset
mba_data <- engineer_features(mba_data)
mba_data$Decided.to.Pursue.MBA <- factor(mba_data$Decided.to.Pursue.MBA, levels = c("No", "Yes"))

# Create feature matrix
mba_features <- mba_data %>%
  select(-matches("Decided.to.Pursue.MBA"), -Person.ID)

# Split data with stratification
set.seed(123)
trainIndex <- createDataPartition(mba_data$Decided.to.Pursue.MBA, p = 0.8, 
                                list = FALSE, times = 1)
trainData <- mba_features[trainIndex, ]
testData <- mba_features[-trainIndex, ]
trainData$Decided.to.Pursue.MBA <- mba_data$Decided.to.Pursue.MBA[trainIndex]
testData$Decided.to.Pursue.MBA <- mba_data$Decided.to.Pursue.MBA[-trainIndex]

# Modified recipe with proper handling of categorical variables
recipe_obj <- recipe(Decided.to.Pursue.MBA ~ ., data = trainData) %>%
  step_dummy(all_nominal_predictors()) %>%
  # Use SMOTE with more balanced parameters
  step_smote(Decided.to.Pursue.MBA, over_ratio = 0.8)

# Prepare the balanced training data
prep_obj <- prep(recipe_obj)
trainData_balanced <- bake(prep_obj, new_data = NULL)

# Modified cross-validation settings
train_control <- trainControl(
  method = "repeatedcv",
  number = 5,  # Reduced from 10 to prevent overfitting
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "down",  # Added down-sampling during CV
  search = "grid"     # Changed to grid search
)

# Define a specific parameter grid for XGBoost
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 4, 5),
  eta = c(0.01, 0.05),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

# Train XGBoost with modified parameters
xgb_model <- train(
  Decided.to.Pursue.MBA ~ .,
  data = trainData_balanced,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  metric = "ROC",
  verbose = FALSE
)
```

    ## [19:49:48] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:49] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:49] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:49] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:50] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:50] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:51] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:51] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:52] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:52] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:53] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:53] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:53] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:53] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:54] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:54] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:55] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:55] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:56] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:56] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:57] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:57] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:58] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:58] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:58] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:58] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:59] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:49:59] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:00] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:00] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:01] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:01] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:01] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:01] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:02] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:02] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:03] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:03] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:04] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:04] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:05] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:05] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:06] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:06] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:07] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:07] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:08] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:08] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:09] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:09] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:10] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:10] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:11] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:11] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:12] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:12] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:13] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:13] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:14] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:14] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:15] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:15] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:16] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:16] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:17] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:17] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:18] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:18] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:18] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:18] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:19] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:19] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:20] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:20] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:21] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:21] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:22] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:22] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:23] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:23] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:23] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:23] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:24] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:24] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:25] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:25] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:26] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:26] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:27] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:27] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:28] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:28] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:28] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:28] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:29] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:29] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:30] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:30] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:31] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:31] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:32] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:32] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:33] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:33] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:33] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:33] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:34] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:34] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:35] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:35] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:36] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:36] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:37] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:37] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:37] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:37] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:38] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:38] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:39] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:39] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:40] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:40] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:41] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:41] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:42] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:42] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:42] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:42] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:43] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:43] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:44] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:44] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:45] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:45] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:46] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:46] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:47] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:47] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:47] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:47] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:48] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:48] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:49] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:49] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:50] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:50] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:51] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:51] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:52] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:52] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:52] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:52] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:53] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:53] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:54] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:54] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:55] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:55] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:56] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:56] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:57] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:57] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:57] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:57] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:58] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:58] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:59] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:50:59] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:00] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:00] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:01] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:01] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:02] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:02] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:02] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:02] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:03] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:03] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:04] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.
    ## [19:51:04] WARNING: src/c_api/c_api.cc:935: `ntree_limit` is deprecated, use `iteration_range` instead.

``` r
# Process test data
test_recipe <- recipe(Decided.to.Pursue.MBA ~ ., data = testData) %>%
  step_dummy(all_nominal_predictors())

test_prep <- prep(test_recipe)
testData_processed <- bake(test_prep, new_data = NULL)

# Make predictions
predictions <- predict(xgb_model, testData_processed)
predictions_prob <- predict(xgb_model, testData_processed, type = "prob")

# Evaluate model
conf_matrix <- confusionMatrix(predictions, testData$Decided.to.Pursue.MBA)
print(conf_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  No Yes
    ##        No  187 252
    ##        Yes 631 929
    ##                                           
    ##                Accuracy : 0.5583          
    ##                  95% CI : (0.5362, 0.5802)
    ##     No Information Rate : 0.5908          
    ##     P-Value [Acc > NIR] : 0.9985          
    ##                                           
    ##                   Kappa : 0.0164          
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##             Sensitivity : 0.22861         
    ##             Specificity : 0.78662         
    ##          Pos Pred Value : 0.42597         
    ##          Neg Pred Value : 0.59551         
    ##              Prevalence : 0.40920         
    ##          Detection Rate : 0.09355         
    ##    Detection Prevalence : 0.21961         
    ##       Balanced Accuracy : 0.50761         
    ##                                           
    ##        'Positive' Class : No              
    ## 

``` r
# ROC curve calculation
library(pROC)
```

    ## Warning: package 'pROC' was built under R version 4.4.2

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
roc_obj <- roc(testData$Decided.to.Pursue.MBA, predictions_prob$Yes)
```

    ## Setting levels: control = No, case = Yes

    ## Setting direction: controls < cases

``` r
auc_value <- auc(roc_obj)
print(paste("AUC:", round(auc_value, 4)))
```

    ## [1] "AUC: 0.5265"

``` r
# Feature importance
importance <- varImp(xgb_model)
print("Top 10 Most Important Features:")
```

    ## [1] "Top 10 Most Important Features:"

``` r
print(importance$importance %>% 
        as.data.frame() %>%
        rownames_to_column("Feature") %>%
        arrange(desc(Overall)) %>%
        head(10))
```

    ##                            Feature   Overall
    ## 1                   GRE.GMAT.Score 100.00000
    ## 2                Undergraduate.GPA  76.46237
    ## 3     Undergraduate.Major_Business  73.49391
    ## 4                      Gender_Male  68.90106
    ## 5   MBA.Funding.Source_Scholarship  64.21969
    ## 6          MBA.Funding.Source_Loan  47.47481
    ## 7    Undergraduate.Major_Economics  36.92216
    ## 8      Undergraduate.Major_Science  34.23514
    ## 9   MBA.Funding.Source_Self.funded  28.76076
    ## 10 Undergraduate.Major_Engineering  27.05750

Model \#2 (Check Claude for previous model, it scored 59%)

``` r
# Load required libraries
library(tidyverse)
library(caret)
library(pROC)
library(car)
```

    ## Warning: package 'car' was built under R version 4.4.2

    ## Loading required package: carData

    ## Warning: package 'carData' was built under R version 4.4.2

    ## 
    ## Attaching package: 'car'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     recode

    ## The following object is masked from 'package:purrr':
    ## 
    ##     some

``` r
# Read the data with specific column types
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

# First, let's verify the data was loaded correctly
print("Data structure after loading:")
```

    ## [1] "Data structure after loading:"

``` r
str(mba_data)
```

    ## 'data.frame':    10000 obs. of  8 variables:
    ##  $ Person.ID             : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ Age                   : int  27 24 33 31 28 33 25 27 30 23 ...
    ##  $ Gender                : chr  "Male" "Male" "Female" "Male" ...
    ##  $ Undergraduate.Major   : chr  "Arts" "Arts" "Business" "Engineering" ...
    ##  $ Undergraduate.GPA     : num  3.18 3.03 3.66 2.46 2.75 3.58 3.06 2.8 2.06 3.51 ...
    ##  $ GRE.GMAT.Score        : int  688 791 430 356 472 409 369 588 521 671 ...
    ##  $ MBA.Funding.Source    : chr  "Loan" "Loan" "Scholarship" "Loan" ...
    ##  $ Decided.to.Pursue.MBA.: chr  "Yes" "No" "No" "No" ...

``` r
# Now convert to factors
mba_data$Gender <- factor(mba_data$Gender)
mba_data$Undergraduate.Major <- factor(mba_data$Undergraduate.Major)
mba_data$MBA.Funding.Source <- factor(mba_data$MBA.Funding.Source)
mba_data$Decided.to.Pursue.MBA. <- factor(mba_data$Decided.to.Pursue.MBA.)

# Create the model matrix
# We'll use model.matrix to handle the categorical variables properly
X <- model.matrix(~ Gender + Undergraduate.Major + MBA.Funding.Source + 
                   Age + Undergraduate.GPA + GRE.GMAT.Score - 1, data = mba_data)

# Create the response variable
y <- factor(mba_data$Decided.to.Pursue.MBA.)

# Combine predictors and response into a single dataframe
model_data <- data.frame(X, MBA_Decision = y)

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(model_data$MBA_Decision, p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Fit logistic regression model
model <- glm(MBA_Decision ~ ., data = train_data, family = "binomial")

# Print model summary
print("Model Summary:")
```

    ## [1] "Model Summary:"

``` r
print(summary(model))
```

    ## 
    ## Call:
    ## glm(formula = MBA_Decision ~ ., family = "binomial", data = train_data)
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                                  Estimate Std. Error z value Pr(>|z|)  
    ## (Intercept)                     4.851e-01  2.546e-01   1.905   0.0568 .
    ## GenderFemale                   -1.042e-01  1.182e-01  -0.881   0.3782  
    ## GenderMale                     -7.384e-02  1.176e-01  -0.628   0.5301  
    ## GenderOther                            NA         NA      NA       NA  
    ## Undergraduate.MajorBusiness    -3.832e-02  7.729e-02  -0.496   0.6200  
    ## Undergraduate.MajorEconomics    4.933e-02  7.674e-02   0.643   0.5203  
    ## Undergraduate.MajorEngineering  1.038e-01  7.772e-02   1.336   0.1815  
    ## Undergraduate.MajorScience      2.822e-02  7.693e-02   0.367   0.7138  
    ## MBA.Funding.SourceLoan         -3.788e-02  6.840e-02  -0.554   0.5797  
    ## MBA.Funding.SourceScholarship  -4.930e-02  6.851e-02  -0.720   0.4717  
    ## MBA.Funding.SourceSelf.funded  -1.033e-01  6.924e-02  -1.491   0.1358  
    ## Age                             1.964e-03  6.043e-03   0.325   0.7452  
    ## Undergraduate.GPA              -3.094e-02  4.264e-02  -0.726   0.4681  
    ## GRE.GMAT.Score                  4.198e-05  1.541e-04   0.272   0.7853  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 9474.2  on 7000  degrees of freedom
    ## Residual deviance: 9466.5  on 6988  degrees of freedom
    ## AIC: 9492.5
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
# Make predictions on test set
predictions_prob <- predict(model, newdata = test_data, type = "response")
predictions_class <- factor(ifelse(predictions_prob > 0.5, "Yes", "No"), 
                          levels = levels(test_data$MBA_Decision))

# Model evaluation metrics
conf_matrix <- confusionMatrix(predictions_class, test_data$MBA_Decision)
print("Confusion Matrix:")
```

    ## [1] "Confusion Matrix:"

``` r
print(conf_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No     0    0
    ##        Yes 1227 1772
    ##                                          
    ##                Accuracy : 0.5909         
    ##                  95% CI : (0.573, 0.6085)
    ##     No Information Rate : 0.5909         
    ##     P-Value [Acc > NIR] : 0.5079         
    ##                                          
    ##                   Kappa : 0              
    ##                                          
    ##  Mcnemar's Test P-Value : <2e-16         
    ##                                          
    ##             Sensitivity : 0.0000         
    ##             Specificity : 1.0000         
    ##          Pos Pred Value :    NaN         
    ##          Neg Pred Value : 0.5909         
    ##              Prevalence : 0.4091         
    ##          Detection Rate : 0.0000         
    ##    Detection Prevalence : 0.0000         
    ##       Balanced Accuracy : 0.5000         
    ##                                          
    ##        'Positive' Class : No             
    ## 

``` r
# Calculate and plot ROC curve
roc_obj <- roc(as.numeric(test_data$MBA_Decision == "Yes"), predictions_prob)
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls > cases

``` r
auc_value <- auc(roc_obj)
print(paste("AUC:", round(auc_value, 3)))
```

    ## [1] "AUC: 0.512"

``` r
# Feature importance based on absolute z-values
coef_summary <- summary(model)$coefficients
importance <- abs(coef_summary[, "z value"])
importance_df <- data.frame(
  Feature = rownames(coef_summary),
  Importance = importance
)
importance_df <- importance_df[order(-importance_df$Importance), ]

print("Top 10 Most Important Features:")
```

    ## [1] "Top 10 Most Important Features:"

``` r
print(head(importance_df, 10))
```

    ##                                                       Feature Importance
    ## (Intercept)                                       (Intercept)  1.9048495
    ## MBA.Funding.SourceSelf.funded   MBA.Funding.SourceSelf.funded  1.4914595
    ## Undergraduate.MajorEngineering Undergraduate.MajorEngineering  1.3360871
    ## GenderFemale                                     GenderFemale  0.8812055
    ## Undergraduate.GPA                           Undergraduate.GPA  0.7256374
    ## MBA.Funding.SourceScholarship   MBA.Funding.SourceScholarship  0.7196538
    ## Undergraduate.MajorEconomics     Undergraduate.MajorEconomics  0.6429045
    ## GenderMale                                         GenderMale  0.6279185
    ## MBA.Funding.SourceLoan                 MBA.Funding.SourceLoan  0.5537814
    ## Undergraduate.MajorBusiness       Undergraduate.MajorBusiness  0.4958336

``` r
# Calculate odds ratios
odds_ratios <- exp(coef(model))
odds_ratios_df <- data.frame(
  Feature = names(odds_ratios),
  OddsRatio = odds_ratios
)

print("Odds Ratios for Top Features:")
```

    ## [1] "Odds Ratios for Top Features:"

``` r
print(head(odds_ratios_df[order(-abs(log(odds_ratios_df$OddsRatio))), ], 10))
```

    ##                                                       Feature OddsRatio
    ## (Intercept)                                       (Intercept) 1.6242775
    ## GenderFemale                                     GenderFemale 0.9010793
    ## Undergraduate.MajorEngineering Undergraduate.MajorEngineering 1.1094183
    ## MBA.Funding.SourceSelf.funded   MBA.Funding.SourceSelf.funded 0.9018820
    ## GenderMale                                         GenderMale 0.9288176
    ## Undergraduate.MajorEconomics     Undergraduate.MajorEconomics 1.0505722
    ## MBA.Funding.SourceScholarship   MBA.Funding.SourceScholarship 0.9518934
    ## Undergraduate.MajorBusiness       Undergraduate.MajorBusiness 0.9624010
    ## MBA.Funding.SourceLoan                 MBA.Funding.SourceLoan 0.9628318
    ## Undergraduate.GPA                           Undergraduate.GPA 0.9695318

``` r
# Save model
saveRDS(model, "mba_decision_model.rds")

# Function for new predictions
predict_mba_probability <- function(new_data) {
  # Ensure new data has same structure as training data
  new_matrix <- model.matrix(~ Gender + Undergraduate.Major + MBA.Funding.Source + 
                              Age + Undergraduate.GPA + GRE.GMAT.Score - 1, 
                            data = new_data)
  predictions <- predict(model, newdata = data.frame(new_matrix), type = "response")
  return(predictions)
}
```
