library(R6)
library(tidyverse)
library(quantreg)
library(sl3)
library(qrnn)

source("R/Lrnr_quantreg.R")
source("R/Lrnr_qrnn.R")
source("R/Lrnr_qgam.R")
source("R/Lrnr_drf.R")

source("R/quantile_tasks.R")

loss_quantile <- function(tau) {
  function(pred, observed) {
    ifelse(observed > pred, tau * abs(observed - pred), (1 - tau) * abs(observed - pred))
  }
}

quantile_sl <- function(task, tau) {
  lightgbm_learner <- Lrnr_lightgbm$new(lightgbm_params = list(objective = "quantile", alpha = tau), num_iterations = 1e1)
  quantreg_learner <- Lrnr_quantreg$new(tau = tau)
  qrnn_learner <- Lrnr_qrnn$new(tau = tau)
  grf_learner <- Lrnr_grf$new(quantiles = tau, quantiles_pred = tau, num.trees = 2e3)
  drf_learner <- Lrnr_drf$new(tau = tau, num.trees = 2e3)
  
  solnp <- Lrnr_solnp$new(eval_function = loss_quantile(tau))
  #sl <- Lrnr_sl$new(learners = list(lightgbm_learner, quantreg_learner, grf_learner, qrnn_learner, drf_learner), solnp)
  sl <- Lrnr_sl$new(learners = list(lightgbm_learner, quantreg_learner), solnp)
  
  options(sl3.verbose = TRUE)
  
  sl_fit <- sl$train(task)
  sl_fit
}

resimulate <- function(task, data, preds, omega, alpha) {
  #data$Eg_simulated <- as.numeric(sn::rsn(n = nrow(data), xi = data$Eg_xi, omega = omega, alpha = alpha))
  #
  #data$Eg_median       <- map_dbl(data$Eg_xi, sn::qsn, p = 0.5, omega = omega, alpha = alpha)
  #data$Eg_quantile_0.1 <- map_dbl(data$Eg_xi, sn::qsn, p = 0.1, omega = omega, alpha = alpha)
  #data$Eg_quantile_0.9 <- map_dbl(data$Eg_xi, sn::qsn, p = 0.9, omega = omega, alpha = alpha)
  #data
}

prediction_fit <- grf::quantile_forest(perovskite_task$X, perovskite_task$Y)
preds <- predict(prediction_fit, quantiles = 0.5)$predictions[,1]
resim <- resimulate(perovskite_task, perovskite, preds, 1, 0)

task <- sl3_Task$new(
  data = resim %>% select(-c(formula, Eg)), #Eg_median, Eg_quantile_0.1, Eg_quantile_0.9, Eg)),
  covariates = setdiff(names(perovskite), c("Eg", "formula", "Eg_simulated", "av_rsp_mean", "av_rsp_std")),
  outcome = "Eg_simulated",
  folds = 5L,
  outcome_type = "continuous"
)

sl <- quantile_sl(task, 0.5)

yhat <- sl$predict()

plot(yhat, task$Y)

plot(sl$learner_fits$Lrnr_grf_2000_0.5_FALSE_NULL_FALSE_0.5_NULL_5_TRUE_0.05_0_1_0.5$predict(), task$Y)
plot(sl$learner_fits$Lrnr_drf_0.5_2000$predict(), task$Y)
plot(sl$learner_fits$Lrnr_lightgbm_1$predict(), task$Y)

fit2 <- grf_learner$train(perovskite_task_formation)
yhat2 <- fit2$predict()

mean(abs(yhat2 - perovskite_task_formation$Y))
