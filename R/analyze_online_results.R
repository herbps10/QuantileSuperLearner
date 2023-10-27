library(dplyr)
library(opera)
library(readr)
library(purrr)

data <- read_rds("./online_simulation_results.rds")

bold <- function(x) {
  x <- paste0("\\textbf{", x, "}")
}

bold_if_best <- function(x) {
  xs <- signif(x, 2)
  xs[xs <= min(xs)] <- bold(xs[xs <= min(xs)])
  xs
}

bold_if_best_coverage <- function(x, alpha) {
  xs <- paste0(signif(x, 3) * 100, "\\%")
  xs[which.min(abs(x - alpha))] <- bold(xs[which.min(abs(x - alpha))])
  xs
}

remove_dups <- function(x) {
  x[x == lag(x)] = ""
  x
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

results <- data %>%
  mutate(loss0.025 = pmap_dbl(list(data, preds0.025, fit0.025), mean_loss, quantile = 0.025),
         loss0.05  = pmap_dbl(list(data, preds0.05,  fit0.05),  mean_loss, quantile = 0.05),
         loss0.1   = pmap_dbl(list(data, preds0.1,   fit0.1),  mean_loss, quantile = 0.1),
         loss0.5   = pmap_dbl(list(data, preds0.5,   fit0.05),  mean_loss, quantile = 0.5),
         loss0.9   = pmap_dbl(list(data, preds0.9,   fit0.9),   mean_loss, quantile = 0.9),
         loss0.95  = pmap_dbl(list(data, preds0.95,  fit0.95),  mean_loss, quantile = 0.95),
         loss0.975 = pmap_dbl(list(data, preds0.975, fit0.975), mean_loss, quantile = 0.975),
         coverage0.8  = pmap_dbl(list(data, preds0.1,   fit0.1, fit0.9), compute_coverage),
         coverage0.9  = pmap_dbl(list(data, preds0.05,  fit0.05, fit0.95), compute_coverage),
         coverage0.95 = pmap_dbl(list(data, preds0.025, fit0.025, fit0.975), compute_coverage),
         width0.8  = pmap_dbl(list(data, preds0.1,   fit0.1, fit0.9), compute_width),
         width0.9  = pmap_dbl(list(data, preds0.05,  fit0.05, fit0.95), compute_width),
         width0.95 = pmap_dbl(list(data, preds0.025, fit0.025, fit0.975), compute_width))

sim_results <- results %>%
  group_by(N, rho, method) %>%
  summarize_at(vars(starts_with("loss"), starts_with("coverage"), starts_with("width")), mean) %>%
  group_by(rho) %>%
  mutate_at(vars(starts_with("loss")), bold_if_best) %>%
  mutate_at(vars(coverage0.8), bold_if_best_coverage, alpha = 0.8) %>%
  mutate_at(vars(coverage0.9), bold_if_best_coverage, alpha = 0.9) %>%
  mutate_at(vars(coverage0.95), bold_if_best_coverage, alpha = 0.95) %>%
  #mutate_at(vars(loss_lower, loss_median, loss_upper, pi_width), bold_if_best) %>%
  #mutate_at(vars(coverage), bold_if_best_coverage) %>%
  mutate(rho = remove_dups(rho))

tab_loss <- sim_results %>%
  select(method, starts_with("loss")) %>%
  knitr::kable(format = "latex", escape = FALSE)

tab_coverage <- sim_results %>%
  select(method, starts_with("coverage")) %>%
  knitr::kable(format = "latex", escape = FALSE)
