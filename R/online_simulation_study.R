library(dplyr)
library(purrr)
library(tidyr)
library(furrr)
library(future)
library(progressr)
library(readr)

simulate <- function(seed, N, dgp = "complex", rho = 0, sigma = 0.1, alpha_lower = 0.1, alpha_upper = 0.9) {
  set.seed(seed)
  N <- N 
  X1 <- runif(N, -2, 2)
  X2 <- runif(N, -2, 2)
  X3 <- runif(N, -2, 2)
  X4 <- runif(N, -2, 2)
  X5 <- runif(N, -2, 2)
  
  if(dgp == "simple") {
    mu <- rep(0, N)
  }
  else {
    mu <- sin(2 * X1) + 0.5 * abs(X2) - 0.5 * X1 * X3 + floor(X4)
  }
  
  epsilon <- numeric(N)
  epsilon[1] <- rnorm(1, 0, sigma / sqrt(1 - rho^2))
  for(n in 2:N) {
    epsilon[n] <- rho * epsilon[n - 1] + rnorm(1, 0, sigma)
  }
  
  data <- tibble(
    t = 1:N,
    X1 = X1,
    X2 = X2,
    X3 = X3,
    X4 = X4,
    X5 = X5,
    mu = mu,
    Y = mu + epsilon,
    q_lower = qnorm(alpha_lower, mu, sigma),
    q_median = qnorm(0.5, mu, sigma),
    q_upper = qnorm(alpha_upper, mu, sigma)
  ) %>%
    mutate(Ylag = lag(Y, 1))
  
  data
}

fit_quantile <- function(data, tau, t0 = 500, p = NULL) {
  if(!is.null(p)) p()
  train <- which(data$t > 1 & data$t <= t0)
  test <- which(data$t > t0)
  
  form1 <- "Y ~ X1 + X2 + X3 + X4 + X5 + Ylag"
  covars <- c("X1", "X2", "X3", "X4", "X5", "Ylag")
  
  x <- as.matrix(data[, covars])
  y <- data$Y
  
  fit1 <- quantreg::rq(form1, data = data[train,], tau = tau, method = "fn")
  fit2 <- lightgbm::lightgbm(x[train,], y[train], params = list(objective = "quantile", alpha = tau), nrounds = 5e2, verbose = -1)
  fit3 <- grf::quantile_forest(x[train,], y[train])
  fit4 <- qrnn::qrnn.fit(x[train,], as.matrix(y[train]), n.hidden = 2, tau = tau)
  fit5 <- qgam::qgam(Y ~ s(X1) + s(X2) + s(X3) + s(X4) + s(X5) + s(Ylag), qu = tau, data = data[train,])
  
  preds <- matrix(nrow = nrow(data), ncol = 5)
  preds[test, 1] <- predict(fit1, newdata = data[test,])
  preds[test, 2] <- predict(fit2, data = x[test, ])
  preds[test, 3] <- predict(fit3, newdata = x[test, ], quantiles = tau)$predictions[,1]
  preds[test, 4] <- qrnn::qrnn.predict(x = x[test, ], parms = fit4)
  preds[test, 5] <- predict(fit5, newdata = data[test,])
  colnames(preds) <- c("quantreg", "lightgbm", "grf", "qrnn", "qgam")
  
  list(preds = preds, test_indices = test)
}

apply_aggregation <- function(data, preds, method, quantile, p = NULL) {
  if(!is.null(p)) p()
  loss.type <- list(name = "pinball", tau = quantile)
  opera::mixture(data[preds$test_indices, ][["Y"]], preds$preds[preds$test_indices, ], model = method, loss.type = loss.type)
}

mean_loss <- function(data, preds, fit, quantile) {
  loss.type <- list(name = "pinball", tau = quantile)
  mean(opera::loss(fit$prediction[, 1], y = data[preds$test_indices, ][["Y"]], loss.type = loss.type))
}

get_predictions <- function(fit) {
  fit$prediction[,1]
}

compute_coverage <- function(data, preds, fit_lower, fit_upper) {
  y <- data$Y[preds$test_indices]
  lower <- fit_lower$prediction[,1]
  upper <- fit_upper$prediction[,1]
  mean(y >= lower & y <= upper)
}

compute_width <- function(data, preds, fit_lower, fit_upper) {
  y <- data$Y[preds$test_indices]
  lower <- fit_lower$prediction[,1]
  upper <- fit_upper$prediction[,1]
  mean(upper - lower)
}


plan(multisession, workers = 40)
N_sims <- 50
setup <- expand_grid(
  sim = 1:N_sims,
  #rho = c(0.5, 0.9, 0.99),
  rho = c(0, 0.5, 0.9, 0.99),
  N = 2e3,
) %>%
  mutate(
    seed = 1:n(),
     data = pmap(list(seed, N, rho), function(seed, N, rho) {
       simulate(seed = seed, N = N, dgp = "complex", rho = rho, sigma = 0.1)
     })
  )

foptions <- furrr::furrr_options(seed = TRUE)

with_progress({
  p <- progressr::progressor(steps = nrow(setup) * 3)
  preds <- setup %>%
    mutate(preds0.025 = future_map(data, fit_quantile, tau = 0.1,   t0 = 1e3, p = p, .options = foptions),
           preds0.05  = future_map(data, fit_quantile, tau = 0.05,  t0 = 1e3, p = p, .options = foptions),
           preds0.1   = future_map(data, fit_quantile, tau = 0.1,   t0 = 1e3, p = p, .options = foptions),
           preds0.5   = future_map(data, fit_quantile, tau = 0.5,   t0 = 1e3, p = p, .options = foptions),
           preds0.9   = future_map(data, fit_quantile, tau = 0.9,   t0 = 1e3, p = p, .options = foptions),
           preds0.95  = future_map(data, fit_quantile, tau = 0.95,  t0 = 1e3, p = p, .options = foptions),
           preds0.975 = future_map(data, fit_quantile, tau = 0.975, t0 = 1e3, p = p, .options = foptions))
})

with_progress({
  p <- progressr::progressor(steps = nrow(preds) * 3)
  
  plan(multisession, workers = 80)
  fits <- preds %>%
    expand_grid(method = c("BOA", "EWA", "SuperLearner")) %>%
    mutate(fit0.025 = future_pmap(list(data, preds0.025, method), apply_aggregation, 0.025, p = p, .options = foptions),
           fit0.05  = future_pmap(list(data, preds0.05, method),  apply_aggregation, 0.05,  p = p, .options = foptions),
           fit0.1   = future_pmap(list(data, preds0.1, method),   apply_aggregation, 0.1,   p = p, .options = foptions),
           fit0.5   = future_pmap(list(data, preds0.5, method),   apply_aggregation, 0.5,   p = p, .options = foptions),
           fit0.9   = future_pmap(list(data, preds0.9, method),   apply_aggregation, 0.9,   p = p, .options = foptions),
           fit0.95  = future_pmap(list(data, preds0.95, method),  apply_aggregation, 0.95,  p = p, .options = foptions),
           fit0.975 = future_pmap(list(data, preds0.975, method), apply_aggregation, 0.975, p = p, .options = foptions))
})

write_rds(fits, "online_simulation_results.rds")
