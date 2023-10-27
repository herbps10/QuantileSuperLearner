library(tidyverse)
library(sl3)
library(R6)

source("quantile_tasks.R")
source("Lrnr_qrnn.R")

loss_quantile <- function(tau) {
  function(pred, observed) {
    pmax(tau * (pred - observed), (tau - 1) * (pred - observed))
  }
}

loss_multi_quantile <- function(taus) {
  losses <- map(taus, loss_quantile)
  function(pred_matrix, observed) {
    preds <- unpack_predictions(pred_matrix)
    reduce(map2(losses, array_branch(preds, 2), rlang::exec, observed = observed, ), `+`)
  }
}

multi_quantile_sl <- function(task, taus) {
  qrnn_learner <- Lrnr_qrnn$new(tau = taus, n.hidden = 2)
  grf_learner <- Lrnr_grf$new(quantiles = taus, quantiles_pred = taus)
  
  solnp <- Lrnr_solnp$new(learner_function = metalearner_linear_multivariate, eval_function = loss_multi_quantile(taus))
  #sl <- Lrnr_sl$new(learners = list(grf_learner, qrnn_learner), solnp)
  sl <- Lrnr_sl$new(learners = list(grf_learner, grf_learner), solnp)
  
  options(sl3.verbose = TRUE)
  
  sl_fit <- sl$train(task)
  sl_fit
}


set.seed(10331)
N <- 1e2
dat <- tibble(
  x = runif(N, -2, 2),
  y = 0.1 * x + rnorm(N, 0, 0.05)
)

covars <- c("x")
task <- sl3_Task$new(
  data = dat,
  covariates = covars,
  outcome = "y",
  folds = 2L
)

taus <- c(0.1, 0.9)
sl <- multi_quantile_sl(task, taus)
cv_results <- cv_sl(sl, task = task, eval_fun = loss_multi_quantile(taus))

preds <- unpack_predictions(sl$predict()) %>%
  as_tibble() %>%
  bind_cols(dat) %>%
  pivot_longer(starts_with("tau"))

ggplot(preds, aes(x = x, y = y)) +
  geom_point() +
  geom_line(aes(y = value, color = name))

taus <- c(0.1, 0.9)
multi_setup <- tribble(
  ~name, ~task,
  "Wine", wine_task,
  "Abalone", abalone_task, 
  "Boston", boston_task
) %>%
  mutate(fit = map(task, multi_quantile_sl, taus = taus),
         cv_fit = pmap(list(fit, task), function(fit, task) {
           cv_sl(lrnr_sl = fit, task = task, eval_fun = loss_multi_quantile(taus))
         }))

cv_risks <- multi_setup %>% 
  mutate(cv_risk = map(cv_fit, `[[`, "cv_risk")) %>%
  select(name, cv_risk) %>%
  unnest(cv_risk) %>%
  mutate(learner = str_replace_all(learner, "Lrnr_", ""),
         learner = str_replace_all(learner, "_.+$", ""),
         learner = str_replace_all(learner, "SuperLearner", "Quantile\nSuperLearner"))

cv_risks %>%
  ggplot(aes(x = risk, y = reorder(learner, risk))) +
  geom_point() +
  facet_wrap(~name, scales = "free") +
  labs(x = "Cross-validated risk", y = "Estimator")
