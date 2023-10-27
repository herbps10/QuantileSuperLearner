library(sl3)
library(tidyverse)
library(R6)
library(data.table)
library(furrr)
library(future)

source("R/Lrnr_quantreg.R")
source("R/Lrnr_qrnn.R")
source("R/Lrnr_qgam.R")
source("R/cv.R")

loss_quantile <- function(tau) {
  function(pred, observed) {
    ifelse(observed > pred, tau * abs(observed - pred), (1 - tau) * abs(observed - pred))
  }
}

loss_interval <- function(taus) {
  alpha <- 1 - (taus[2] - taus[1])
  function(pred, observed) {
    pred <- unpack_predictions(pred)
    pred[,2] - pred[,1] + 2 / alpha * (pred[,1] - observed) * (observed < pred[,1]) + 2/alpha * (observed - pred[,2]) * (observed > pred[,2])
  }
}

quantile_sl_interval <- function(task, taus) {
  qrnn_learner <- Lrnr_qrnn$new(tau = taus, n.hidden = 2)
  qrnn_learner2 <- Lrnr_qrnn$new(tau = taus, n.hidden = 3)
  grf_learner <- Lrnr_grf$new(quantiles = taus, quantiles_pred = taus, num.trees = 2e3)
  
  solnp <- Lrnr_solnp$new(learner_function = metalearner_linear_multivariate, eval_function = loss_interval(taus))
  sl <- Lrnr_sl$new(learners = list(qrnn_learner, qrnn_learner2, grf_learner), solnp)
  
  sl_fit <- sl$train(task)
  sl_fit
}

quantile_sl <- function(task, tau) {
  lightgbm_learner <- Lrnr_lightgbm$new(obj = "quantile", objective = "quantile", alpha = tau, nrounds = 150, verbose = -1)
  quantreg_learner <- Lrnr_quantreg$new(tau = tau)
  #quantreg_lasso_learner <- Lrnr_quantreg$new(tau = tau, method = "lasso")
  qrnn_learner <- Lrnr_qrnn$new(tau = tau, n.hidden = 2)
  #qrnn_learner2 <- Lrnr_qrnn$new(tau = tau, n.hidden = 3)
  grf_learner <- Lrnr_grf$new(quantiles = tau, quantiles_pred = tau, num.trees = 2e3)
  #xgboost_learner <- Lrnr_xgboost$new(objective = "reg:quantileerror", quantile_alpha = tau, learning_rate = 0.1, nrounds = 200, max_depth = 1, subsample = 0.5)
  qgam_learner <- Lrnr_qgam$new(qu = tau)
  
  solnp <- Lrnr_solnp$new(eval_function = loss_quantile(tau))
  #sl <- Lrnr_sl$new(learners = list(quantreg_learner, lightgbm_learner, xgboost_learner, grf_learner, qrnn_learner, qrnn_learner2, quantreg_lasso_learner, qgam_learner), solnp)
  sl <- Lrnr_sl$new(learners = list(quantreg_learner, lightgbm_learner, qrnn_learner, grf_learner, qgam_learner), solnp)
  
  sl_fit <- sl$train(task)
  sl_fit
}

simulate <- function(seed, N, dgp, holdout_frac = 0.2, alpha_lower = 0.1, alpha_upper = 0.9) {
  set.seed(seed)
  N <- N + 1e3
  X1 <- runif(N, -2, 2)
  X2 <- runif(N, -2, 2)
  X3 <- runif(N, -2, 2)
  X4 <- runif(N, -2, 2)
  X5 <- runif(N, -2, 2)
  
  if(dgp == "simple") {
    mu <- sin(X1)
  }
  else {
    mu <- sin(2 * X1) + 0.5 * abs(X2) - 0.5 * X1 * X3 + floor(X4)
  }
  
  sigma <- 0.1
  
  data <- tibble(
    X1 = X1,
    X2 = X2,
    X3 = X3,
    X4 = X4,
    X5 = X5,
    mu = mu,
    Y = mu + rnorm(N, 0, sigma),
    q_lower = qnorm(alpha_lower, mu, sigma),
    q_median = qnorm(0.5, mu, sigma),
    q_upper = qnorm(alpha_upper, mu, sigma),
    heldout = FALSE
  )
  heldout <- sample(1:nrow(data), size = 1e3)
  data$heldout[heldout] <- TRUE
  
  data
}

fit <- function(data) {
  task <- sl3_Task$new(
    data = data[data$heldout == FALSE,],
    covariates = c("X1", "X2", "X3", "X4", "X5"),
    outcome = "Y",
    folds = 5L,
    outcome_type = "continuous"
  )
  
  fit_lower <- quantile_sl(task, 0.1)
  fit_median <- quantile_sl(task, 0.5)
  fit_upper <- quantile_sl(task, 0.9)
  #fit_interval <- quantile_sl_interval(task, c(0.1, 0.9))
  
  holdout_data <- data[data$heldout == TRUE, ]
  
  holdout_task <- sl3_Task$new(
    data = holdout_data,
    covariates = c("X1", "X2", "X3", "X4", "X5"),
    outcome = "Y",
    outcome_type = "continuous"
  )
      
  sl_lower  <- fit_lower$predict(holdout_task)
  sl_median <- fit_median$predict(holdout_task)
  sl_upper  <- fit_upper$predict(holdout_task)
  
  preds <- tibble(
    learner = rep("SuperLearner", length(sl_lower)), 
    index = 1:length(sl_lower), 
    lower = sl_lower, 
    median = sl_median, 
    upper = sl_upper
  )
  
  for(n in seq_along(fit_lower$learner_fits)) {
    learner_name <- str_replace_all(names(fit_lower$learner_fits)[[n]], "_0\\.[159]", "")
    preds <- bind_rows(preds, tibble(
      index = 1:length(sl_lower),
      learner = rep(learner_name, length(sl_lower)),
      lower = fit_lower$learner_fits[[n]]$predict(holdout_task),
      median = fit_median$learner_fits[[n]]$predict(holdout_task),
      upper = fit_upper$learner_fits[[n]]$predict(holdout_task)
    ))
  }
  preds <- preds %>%
    left_join(mutate(holdout_data, index = 1:n())) %>%
    select(-index)
  
  gc()
  
  return(list(
    preds = preds,
    coefs = list(lower = fit_lower$coefficients, median = fit_median$coefficients, upper = fit_upper$coefficients)
  ))
}

foptions <- furrr_options(packages = c("sl3", "dplyr", "R6", "data.table"), seed = TRUE)

plan(multisession, workers = 4)

set.seed(1553)
setup <- expand_grid(
  index = 1:1,
  #N = c(250, 500, 1000),
  N = c(250, 500, 1000),
  #dgp = c("simple", "complex")
  dgp = "complex",
) %>%
  mutate(seed = 1:n(),
         data = pmap(list(seed, N, dgp), simulate))

results <- setup %>%
  mutate(fit = future_map(data, fit, .options = foptions))

write_rds(results, "iid_simulation_results.rds")
