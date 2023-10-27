library(sl3)
library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(stringr)
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
  qrnn_learner <- Lrnr_qrnn$new(tau = tau, n.hidden = 2)
  grf_learner <- Lrnr_grf$new(quantiles = tau, quantiles_pred = tau, num.trees = 2e3)
  qgam_learner <- Lrnr_qgam$new(qu = tau)
  
  solnp <- Lrnr_solnp$new(eval_function = loss_quantile(tau))
  sl <- Lrnr_sl$new(learners = list(quantreg_learner, lightgbm_learner, qrnn_learner, grf_learner, qgam_learner), solnp)
  
  sl_fit <- sl$train(task)
  sl_fit
}

simulate <- function(seed, N, dgp, holdout_frac = 0.2) {
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
    q0.025 = qnorm(0.025, mu, sigma),
    q0.05 = qnorm(0.05, mu, sigma),
    q0.1 = qnorm(0.1, mu, sigma),
    q0.5 = qnorm(0.5, mu, sigma),
    q0.9 = qnorm(0.9, mu, sigma),
    q0.95 = qnorm(0.95, mu, sigma),
    q0.975 = qnorm(0.975, mu, sigma),
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

  fit0.025 <- quantile_sl(task, 0.025)
  fit0.05  <- quantile_sl(task, 0.05)
  fit0.1   <- quantile_sl(task, 0.1)
  fit0.5   <- quantile_sl(task, 0.5)
  fit0.9   <- quantile_sl(task, 0.9)
  fit0.95  <- quantile_sl(task, 0.95)
  fit0.975 <- quantile_sl(task, 0.975)
  
  holdout_data <- data[data$heldout == TRUE, ]
  
  holdout_task <- sl3_Task$new(
    data = holdout_data,
    covariates = c("X1", "X2", "X3", "X4", "X5"),
    outcome = "Y",
    outcome_type = "continuous"
  )
      
  sl0.025  <- fit0.025$predict(holdout_task)
  sl0.05   <- fit0.05$predict(holdout_task)
  sl0.1    <- fit0.1$predict(holdout_task)
  sl0.5    <- fit0.5$predict(holdout_task)
  sl0.9    <- fit0.9$predict(holdout_task)
  sl0.95   <- fit0.95$predict(holdout_task)
  sl0.975  <- fit0.975$predict(holdout_task)
  
  preds <- tibble(
    learner   = rep("SuperLearner", length(sl0.1)), 
    index     = 1:length(sl0.1), 
    pred0.025 = sl0.025, 
    pred0.05  = sl0.05, 
    pred0.1   = sl0.1, 
    pred0.5   = sl0.5, 
    pred0.9   = sl0.9,
    pred0.95  = sl0.95,
    pred0.975 = sl0.975
  )
  
  for(n in seq_along(fit0.1$learner_fits)) {
    learner_name <- str_replace_all(names(fit0.1$learner_fits)[[n]], "_0\\.[159]", "")
    preds <- bind_rows(preds, tibble(
      index = 1:length(sl0.1),
      learner = rep(learner_name, length(sl0.1)),
      pred0.025 = fit0.025$learner_fits[[n]]$predict(holdout_task),
      pred0.05  = fit0.05$learner_fits[[n]]$predict(holdout_task),
      pred0.1   = fit0.1$learner_fits[[n]]$predict(holdout_task),
      pred0.5   = fit0.5$learner_fits[[n]]$predict(holdout_task),
      pred0.9   = fit0.9$learner_fits[[n]]$predict(holdout_task),
      pred0.95  = fit0.95$learner_fits[[n]]$predict(holdout_task),
      pred0.975 = fit0.975$learner_fits[[n]]$predict(holdout_task)
    ))
  }
  preds <- preds %>%
    left_join(mutate(holdout_data, index = 1:n())) %>%
    select(-index)
  
  gc()
  
  return(list(
    preds = preds,
    coefs = list(
      "0.025" = fit0.025$coefficients, 
      "0.05"  = fit0.05$coefficients, 
      "0.1"   = fit0.1$coefficients, 
      "0.5"   = fit0.5$coefficients, 
      "0.9"   = fit0.9$coefficients, 
      "0.95"  = fit0.95$coefficients, 
      "0.975" = fit0.975$coefficients
    )
  ))
}

foptions <- furrr_options(packages = c("sl3", "dplyr", "R6", "data.table"), seed = TRUE)

plan(multisession, workers = 40)

set.seed(1553)
N_sims <- 50
setup <- expand_grid(
  index = 1:N_sims,
  N = c(250, 500, 1000),
  dgp = "complex",
) %>%
  mutate(seed = 1:n(),
         data = pmap(list(seed, N, dgp), simulate))

results <- setup %>%
  mutate(fit = future_map(data, fit, .options = foptions))

write_rds(results, "iid_simulation_results.rds")
