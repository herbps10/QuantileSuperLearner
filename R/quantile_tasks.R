library(sl3)
library(tidyverse)

loss_quantile <- function(tau) {
  function(pred, observed) {
    ifelse(observed > pred, tau * abs(observed - pred), (1 - tau) * abs(observed - pred))
  }
}

quantile_sl <- function(task, tau) {
  lightgbm_learner <- Lrnr_lightgbm$new(obj = "quantile", objective = "quantile", alpha = tau, nrounds = 100)
  quantreg_learner <- Lrnr_quantreg$new(tau = tau)
  qrnn_learner <- Lrnr_qrnn$new(tau = tau)
  grf_learner <- Lrnr_grf$new(quantiles = tau, quantiles_pred = tau, num.trees = 2e3)
  drf_learner <- Lrnr_drf$new(tau = tau, num.trees = 2e3)
  gbm_learner <- Lrnr_gbm$new(distribution = list(name = "quantile", alpha = tau), interaction.depth = 2, shrinkage = 0.1)
  
  solnp <- Lrnr_solnp$new(eval_function = loss_quantile(tau))
  sl <- Lrnr_sl$new(learners = list(lightgbm_learner, quantreg_learner, grf_learner, qrnn_learner, drf_learner), cv_control = list(V = 5), solnp)
  
  options(sl3.verbose = TRUE)
  
  sl_fit <- sl$train(task)
  sl_fit
}

perovskite <- read_csv(
  "data/combine.csv"
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
  covariates = setdiff(names(perovskite), c("formula", "Eg", "Ef", "av_rsp_mean", "av_rsp_std")),
  outcome = "Ef",
  folds = 5L,
  outcome_type = "continuous"
)

