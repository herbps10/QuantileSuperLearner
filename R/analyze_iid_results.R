library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(purrr)

source("R/quantile_tasks.R")

losses <- lapply(c(0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975), loss_quantile)

learner_titles <- tribble(
	~learner, ~title,
	"Lrnr_lightgbm_1_quantile_quantile_150_-1", "GBM (\\texttt{lightgbm})",
	"Lrnr_quantreg", "QReg",
	"Lrnr_grf_2000_FALSE_NULL_FALSE_NULL_5_TRUE_0.05_0_1", "GRF",
	"Lrnr_qrnn_2", "QRNN",
	"Lrnr_drf_2000", "DRF",
	"Lrnr_qgam_NULL_NULL_GCV.Cp", "QGAM",
	"SuperLearner", "Quantile SuperLearner"
)

data <- read_rds("./iid_simulation_results.rds")

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

results <- data %>%
  mutate(preds = map(fit, `[[`, "preds")) %>% 
  select(index, N, dgp, preds) %>% 
  unnest(c(preds)) %>%
  mutate(
    loss0.025    = losses[[1]](pred0.025, Y),
    loss0.05     = losses[[2]](pred0.05,  Y),
    loss0.1      = losses[[3]](pred0.1,   Y),
    loss0.5      = losses[[4]](pred0.5,   Y),
    loss0.9      = losses[[5]](pred0.9,   Y),
    loss0.95     = losses[[6]](pred0.95,  Y),
    loss0.975    = losses[[7]](pred0.975, Y),
    coverage0.8  = pred0.1 <= Y & pred0.9 >= Y,
    coverage0.9  = pred0.05 <= Y & pred0.95 >= Y,
    coverage0.95 = pred0.025 <= Y & pred0.975 >= Y,
    pi_width0.8  = pred0.9 - pred0.1,
    pi_width0.9  = pred0.95 - pred0.05,
    pi_width0.95 = pred0.975 - pred0.025
  ) %>%
  group_by(N, learner) %>%
  summarize_at(vars(starts_with("loss"), starts_with("coverage"), starts_with("pi_width")), mean) %>%
  left_join(learner_titles) %>%
  #select(title, loss_lower, loss_median, loss_upper, coverage, pi_width) %>%
  group_by(N) %>%
  mutate_at(vars(starts_with("loss")), bold_if_best) %>%
  mutate_at(vars(coverage0.8), bold_if_best_coverage, alpha = 0.8) %>%
  mutate_at(vars(coverage0.9), bold_if_best_coverage, alpha = 0.9) %>%
  mutate_at(vars(coverage0.95), bold_if_best_coverage, alpha = 0.95) %>%
  #mutate_at(vars(loss_lower, loss_median, loss_upper, pi_width), bold_if_best) %>%
  #mutate_at(vars(coverage), bold_if_best_coverage) %>%
  mutate(N = remove_dups(N))

tab_loss <- results %>%
  select(title, starts_with("loss")) %>%
  knitr::kable(format = "latex", escape = FALSE)

tab_coverage <- results %>%
  select(title, starts_with("coverage")) %>%
  knitr::kable(format = "latex", escape = FALSE)
