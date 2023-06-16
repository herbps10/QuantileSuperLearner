library(R6)
library(tidyverse)
library(quantreg)
library(sl3)
library(qrnn)
library(data.table)

source("R/Lrnr_quantreg.R")
source("R/Lrnr_qrnn.R")
source("R/Lrnr_qgam.R")
source("R/Lrnr_drf.R")

source("R/quantile_tasks.R")

source("R/cv.R")

quantiles <- c(0.1, 0.5, 0.9)
setup <- tribble(
  ~name, ~task,
  "Energy Formation", perovskite_task_formation,
  "Energy Bandgap", perovskite_task
) %>%
  expand_grid(quantile = quantiles) %>%
  mutate(fit = map2(task, quantile, quantile_sl),
         cv_fit = pmap(list(fit, task, quantile), function(fit, task, quantile) {
           cv_sl(lrnr_sl = fit, task = task, eval_fun = loss_quantile(quantile))
         }))

#lower <- setup$cv_fit[[1]]$cv_sl_fit$predict()
#upper <- setup$cv_fit[[2]]$cv_sl_fit$predict()

#plot(lower, upper)

#lower <- setup$fit[[1]]$predict()
#upper <- setup$fit[[2]]$predict()
#mean(lower <= perovskite_task$Y & upper >= perovskite_task$Y)
#
#mean(
#  setup$fit[[1]]$learner_fits$Lrnr_gbm_10000_2_0.1$predict() <= perovskite_task$Y &
#  setup$fit[[2]]$learner_fits$Lrnr_gbm_10000_2_0.1$predict() >= perovskite_task$Y
#)

write_rds(setup, "cv_results.rds")


