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

quantiles <- c(0.1, 0.5, 0.9)
setup <- tribble(
  ~name, ~task,
  "Energy Bandgap", perovskite_task
) %>%
  expand_grid(quantile = quantiles) %>%
  mutate(fit = map2(task, quantile, quantile_sl),
         cv_fit = pmap(list(fit, task, quantile), function(fit, task, quantile) {
           cv_sl(lrnr_sl = fit, task = task, eval_fun = loss_quantile(quantile))
         }))

write_rds(setup, "cv_results.rds")


