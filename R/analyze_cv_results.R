library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(stringr)

data <- read_rds("cv_results.rds")

coverage <- data %>%
  mutate(Y = map(task, function(task) as_tibble(task$X) %>% mutate(Y = task$Y))) %>% 
  mutate(sl_cv_preds = map(fit, function(fit) {
    tibble(value = fit$cv_fit$preds)
  })) %>%
  select(name, quantile, sl_cv_preds, Y) %>%
  unnest(c(sl_cv_preds, Y)) %>%
  tidyr::pivot_wider(names_from = "quantile", values_from = "value") %>%
  mutate(covered0.8 = `0.1` <= Y & `0.9` >= Y,
         covered0.9 = `0.05` <= Y & `0.95` >= Y,
         covered0.95 = `0.025` <= Y & `0.975` >= Y,
         pi_width0.8 = `0.9` - `0.1`,
         pi_width0.9 = `0.95` - `0.05`,
         pi_width0.95 = `0.975` - `0.025`) %>%
  group_by(name) %>%
  summarize(coverage0.8 = mean(covered0.8),
            coverage0.9 = mean(covered0.9),
            coverage0.95 = mean(covered0.95),
            pi_width0.8 = mean(pi_width0.8),
            pi_width0.9 = mean(pi_width0.9),
            pi_width0.95 = mean(pi_width0.95)
  ) %>%
	mutate(learner = "SuperLearner")

candidate_coverages <- data %>%
  mutate(Y = map(task, function(task) tibble(Y = task$Y, index = 1:length(task$Y))),
	 candidate_preds = map(fit, function(fit) fit$cv_fit[["candidate_preds"]])) %>% 
  select(name, quantile, candidate_preds, Y) %>%
  mutate(candidate_preds = map(candidate_preds, function(candidate_preds) {
    preds <- as_tibble(candidate_preds)
		colnames(preds) <- str_replace_all(colnames(preds), "_0\\.[021579]{1,3}", "")
		preds
  })) %>%
  unnest(c(candidate_preds, Y)) %>%
  pivot_longer(starts_with("Lrnr"), names_to = "learner") %>%
  tidyr::pivot_wider(names_from = "quantile", values_from = "value") %>%
  mutate(
    covered0.8 = `0.1` <= Y & `0.9` >= Y, 
    covered0.9 = `0.05` <= Y & `0.95` >= Y, 
    covered0.95 = `0.025` <= Y & `0.975` >= Y, 
    pi_width0.8 = `0.9` - `0.1`,
    pi_width0.9 = `0.95` - `0.05`,
    pi_width0.95 = `0.975` - `0.025`
  ) %>%
  group_by(name, learner) %>%
  summarize(coverage0.8 = mean(covered0.8), 
            coverage0.9 = mean(covered0.9), 
            coverage0.95 = mean(covered0.95), 
            pi_width0.8 = mean(pi_width0.8),
            pi_width0.9 = mean(pi_width0.9),
            pi_width0.95 = mean(pi_width0.95)
  )

tab1_data <- data %>% 
  mutate(cv_risk = map(fit, function(fit) as_tibble(fit$cv_fit$cv_risk))) %>%
  select(name, quantile, cv_risk) %>% 
  unnest(c(cv_risk)) %>%
  mutate(learner = if_else(learner == "SuperLearner", learner, str_replace_all(learner, "_0\\.[012579]{1,3}", ""))) %>%
  select(name, learner, quantile, risk) %>%
  pivot_wider(names_from = c("quantile"), values_from = "risk")

bold <- function(x) {
  x <- paste0("\\textbf{", x, "}")
}

bold_if_best <- function(x) {
  xs <- signif(x, 2)
  indices <- which(xs == min(xs))
  xs[indices] <- bold(xs[indices])
  xs
}

bold_if_best_coverage <- function(x, level) {
  xs <- paste0(signif(x, 3) * 100, "\\%")
  indices <- which(abs(signif(x, 3) - level) == min(abs(signif(x, 3) - level)))
  xs[indices] <- bold(xs[indices])
  xs
}

learner_titles <- tribble(
	~learner, ~title,
	"Lrnr_lightgbm_1_quantile_quantile_100", "GBM (\\texttt{lightgbm})",
	"Lrnr_quantreg_fn", "QReg",
	"Lrnr_grf_2000_FALSE_NULL_FALSE_NULL_5_TRUE_0_1", "GRF",
	"Lrnr_qrnn_2", "QRNN 2",
	"Lrnr_qrnn_3", "QRNN 3",
	"Lrnr_drf_2000", "DRF",
	"SuperLearner", "Quantile SuperLearner"
)

tab1 <- tab1_data %>%
	left_join(learner_titles) %>%
  select(-learner) %>%
  select(name, title, `0.025`, `0.05`, `0.1`, `0.5`, `0.9`, `0.95`, `0.975`) %>%
  group_by(name) %>%
  mutate_if(is.numeric, bold_if_best) %>%
  arrange(name, title)

tab2 <- bind_rows(coverage, candidate_coverages) %>%
	left_join(learner_titles) %>%
	select(-learner) %>%
	select(name, title, coverage0.8, coverage0.9, coverage0.95) %>%
  group_by(name) %>%
	mutate_at(vars(coverage0.8), bold_if_best_coverage, 0.8) %>%
	mutate_at(vars(coverage0.9), bold_if_best_coverage, 0.9) %>%
	mutate_at(vars(coverage0.95), bold_if_best_coverage, 0.95) %>%
  ungroup() %>%
  arrange(name, title) 
