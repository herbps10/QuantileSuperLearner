library(sl3)
library(dplyr)
library(readr)

loss_quantile <- function(tau) {
  function(pred, observed) {
    pmax(tau * (pred - observed), (tau - 1) * (pred - observed))
  }
}

quantile_sl <- function(task, tau) {
  lightgbm_learner <- Lrnr_lightgbm$new(lightgbm_params = list(objective = "quantile", alpha = tau, nrounds = 100), nrounds = 100)
  quantreg_learner <- Lrnr_quantreg$new(tau = tau)
  qrnn_learner <- Lrnr_qrnn$new(tau = tau)
  grf_learner <- Lrnr_grf$new(quantiles = tau, quantiles_pred = tau, num.trees = 2e3)
  drf_learner <- Lrnr_drf$new(tau = tau, num.trees = 2e3)
  gbm_learner <- Lrnr_gbm$new(distribution = list(name = "quantile", alpha = tau))
  
  solnp <- Lrnr_solnp$new(eval_function = loss_quantile(tau))
  #sl <- Lrnr_sl$new(learners = list(lightgbm_learner, quantreg_learner, grf_learner, qrnn_learner, drf_learner), solnp)
  sl <- Lrnr_sl$new(learners = list(quantreg_learner, lightgbm_learner, grf_learner, drf_learner), cv_control = list(V = 5), solnp)
  
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
