library(R6)
library(dplyr)
library(tibble)
library(quantreg)
library(sl3)
library(qrnn)
library(tidyr)
library(purrr)
library(data.table)
library(future)
library(furrr)

print("Loading files...")
source("R/Lrnr_quantreg.R")
source("R/Lrnr_qrnn.R")
source("R/Lrnr_qgam.R")
source("R/Lrnr_drf.R")

source("R/quantile_tasks.R")

source("R/cv.R")
print("Done loading")

plan(multisession, workers = 28)

quantiles <- c(0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975)
opts <- furrr_options(seed = TRUE, globals = c("my_cv_sl", "quantile_sl", "loss_quantile", "Lrnr_quantreg", "Lrnr_qrnn", "Lrnr_qgam", "Lrnr_drf"), packages = c("R6", "sl3", "qrnn", "qgam", "drf", "gbm", "grf", "tibble", "dplyr", "tidyr", "purrr", "data.table"))

options(sl3.verbose = TRUE)

setup <- tribble(
  ~name, ~task,
  "Energy Formation", perovskite_task_formation,
  "Energy Bandgap", perovskite_task
) %>%
  expand_grid(quantile = quantiles) %>%
  mutate(fit = future_map2(task, quantile, function(task, quantile) {
    library(R6)
    source("R/Lrnr_quantreg.R")
    source("R/Lrnr_qrnn.R")
    source("R/Lrnr_qgam.R")
    source("R/Lrnr_drf.R")

    print(glue::glue("Starting fit for {quantile}..."))

    fit = quantile_sl(task, quantile)
    preds = as_tibble(fit$predict())
    preds_cv = as_tibble(fit$fit_object$cv_fit$predict())

    options(sl3.verbose = TRUE)

    print(glue::glue("Starting cv fit for {quantile}"))
    cv_fit = my_cv_sl(lrnr_sl = fit, task = task, eval_fun = loss_quantile(quantile))

    list(preds = preds, preds_cv = preds_cv, cv_fit = cv_fit)
  #}))
  }, .options = opts))

write_rds(setup, "cv_results.rds")


