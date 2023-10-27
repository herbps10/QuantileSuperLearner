library(dplyr)
library(opera)
library(readr)
library(purrr)

fits <- read_rds("solar_results.rds")

mean_loss <- function(data, preds, fit, quantile) {
  loss.type <- list(name = "pinball", tau = quantile)
  mean(opera::loss(fit$prediction[, 1], y = data[preds$test_indices, ][["NSRDB_GHI"]], loss.type = loss.type))
}

get_predictions <- function(fit) {
  fit$prediction[,1]
}

compute_coverage <- function(data, test_indices, fit_lower, fit_upper) {
  y <- data$NSRDB_GHI[test_indices]
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

results <- fits %>%
  mutate(loss = pmap_dbl(list(data, preds, fit, quantile), mean_loss))


bold <- function(x) {
  x <- paste0("\\textbf{", x, "}")
}

bold_if_best <- function(x) {
  xs <- signif(x, 3)
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

tab_alt <- results %>% 
  select(id, quantile, method, loss) %>% 
  pivot_wider(names_from = "id", values_from = "loss") %>% 
  arrange(quantile, method) %>% 
  group_by(quantile) %>%
  mutate_at(vars(BON:TBL), bold_if_best) %>% 
  mutate(quantile = remove_dups(quantile)) %>%
  mutate(method = ifelse(method == "SuperLearner", "QSL", method)) %>%
  knitr::kable(format = "latex", escape = FALSE)


results_coverage <- fits %>%
  mutate(test_indices = map(preds, `[[`, "test_indices")) %>%
  select(-preds) %>%
  pivot_wider(values_from = c("fit"), names_from = c("quantile")) %>%
  mutate(coverage0.8  = pmap_dbl(list(data, test_indices, `0.1`, `0.9`), compute_coverage),
         coverage0.9  = pmap_dbl(list(data, test_indices, `0.05`, `0.95`), compute_coverage),
         coverage0.95 = pmap_dbl(list(data, test_indices, `0.025`, `0.975`), compute_coverage)) %>%
  select(id, method, starts_with("coverage"))  %>%
  arrange(id, method) 

tab_coverage <- results_coverage %>%
  group_by(id) %>%
  mutate_at(vars(coverage0.8), bold_if_best_coverage, alpha = 0.8) %>%
  mutate_at(vars(coverage0.9), bold_if_best_coverage, alpha = 0.9) %>%
  mutate_at(vars(coverage0.95), bold_if_best_coverage, alpha = 0.95) %>%
  ungroup() %>%
  mutate(id = remove_dups(id)) %>%
  mutate(method = ifelse(method == "SuperLearner", "QSL", method)) %>%
  knitr::kable(format = "latex", escape = FALSE)
