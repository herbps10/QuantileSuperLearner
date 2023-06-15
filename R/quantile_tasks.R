library(sl3)
library(tidyverse)


loss_quantile <- function(tau) {
  function(pred, observed) {
    pmax(tau * (pred - observed), (tau - 1) * (pred - observed))
  }
}

quantile_sl <- function(task, tau) {
  lightgbm_learner <- Lrnr_lightgbm$new(lightgbm_params = list(objective = "quantile", alpha = tau))
  quantreg_learner <- Lrnr_quantreg$new(tau = tau)
  qrnn_learner <- Lrnr_qrnn$new(tau = tau)
  grf_learner <- Lrnr_grf$new(quantiles = tau, quantiles_pred = tau, num.trees = 2e3)
  drf_learner <- Lrnr_drf$new(tau = tau, num.trees = 2e3)
  
  solnp <- Lrnr_solnp$new(eval_function = loss_quantile(tau))
  #sl <- Lrnr_sl$new(learners = list(lightgbm_learner, quantreg_learner, grf_learner, qrnn_learner, drf_learner), solnp)
  sl <- Lrnr_sl$new(learners = list(lightgbm_learner, quantreg_learner, grf_learner, drf_learner), solnp)
  
  options(sl3.verbose = TRUE)
  
  sl_fit <- sl$train(task)
  sl_fit
}

# Read abalone data
abalone <- read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", col_names = c("sex", "length", "diameter", "height", "weight.whole", "weight.shucked", 
                    "weight.viscera", "weight.shell", "rings"))

winequality <- read_delim("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", delim = ";")
names(winequality) <- str_replace_all(names(winequality), " ", "")


concrete <- read_csv(
  "data/concrete+compressive+strength/Concrete_Data.csv", 
  col_names = c("cement", "slag", "flyash", "water", "superplasticizer", "coarse", "fine", "age", "strength"),
  skip = 1
)

perovskite <- read_csv(
  "data/combine.csv"
)

boston_task <- sl3_Task$new(
  data = MASS::Boston,
  covariates = c("crim", "zn", "indus", "chas", "nox", "age", "dis", "rad", "tax", "ptratio", "black", "lstat"),
  outcome = "medv",
  folds = 5L,
  outcome_type = "continuous"
)

abalone_task <- sl3_Task$new(
  data = abalone,
  covariates = c("sex", "length", "diameter", "height", "weight.whole", "weight.shucked", "weight.viscera", "weight.shell"),
  outcome = "rings",
  folds = 5L,
  outcome_type = "continuous"
)

wine_task <- sl3_Task$new(
  data = winequality,
  covariates = c("fixedacidity", "volatileacidity", "citricacid", "residualsugar", "chlorides",
                 "freesulfurdioxide", "totalsulfurdioxide", "density", "pH", "sulphates",
                 "alcohol"),
  outcome = "quality",
  folds = 5L,
  outcome_type = "continuous"
)

concrete_task <- sl3_Task$new(
  data = concrete,
  covariates = c("cement", "slag", "flyash", "water", "superplasticizer", "coarse", "fine", "age"),
  outcome = "strength",
  folds = 5L,
  outcome_type = "continuous"
)


perovskite_task <- sl3_Task$new(
  data = perovskite,
  covariates = setdiff(names(perovskite), c("formula", "Eg", "av_rsp_mean", "av_rsp_std")),
  outcome = "Eg",
  folds = 5L,
  outcome_type = "continuous"
)

perovskite_task_formation <- sl3_Task$new(
  data = perovskite,
  #covariates = setdiff(names(perovskite), c("formula", "Eg", "Ef", "av_rsp_mean", "av_rsp_std")),
  covariates <- c("z_mean", "z_std"),
  outcome = "Ef",
  folds = 5L,
  outcome_type = "continuous"
)
