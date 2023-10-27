library(sl3)
library(tidyverse)
library(SuperLearner)
library(origami)
library(R6)
library(furrr)

source("Lrnr_quantreg.R")
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

curve(loss_quantile(0.1)(x, 0), -1, 1)

loss_coverage <- function(coverage) {
  function(pred_matrix, observed) {
    preds <- unpack_predictions(pred_matrix)
    within = preds[,1] <= observed & preds[,2] >= observed
    return((within - coverage)^2)
    #return(within)
  }
}

make_task <- function(set) {
  sl3_Task$new(
    data = X[set,],
    covariates = covars,
    outcome = "log_n"
  )
}

train_base_learners <- function(task) {
  list(
    #quantreg = quantreg_learner$train(task),
    grf = grf_learner$train(task),
    qrnn = qrnn_learner$train(task)
  )
}

evaluate_performance <- function(learners, validation_task, loss) {
  map(learners, function(learner) loss(learner$predict(validation_task), validation_task$Y))
  #map(learners, function(learner) loss_coverage(taus[2] - taus[1])(learner$predict(validation_task), validation_task$Y))
}

learner_predictions <- function(base_learners, task) {
  predictions <- map(base_learners, function(learner) learner$predict(task)[,1])
  predictions$Y <- task$Y
  predictions <- as_tibble(predictions)
  names(predictions) <- c(names(base_learners), "Y")
  predictions
}

make_sl_task <- function(base_learners, training_data) {
  sl3_Task$new(
    data = training_data,
    covariates = setdiff(names(base_learners), "Y"),
    outcome = "Y"
  )
}

make_sl <- function(meta_task) {
  solnp$train(meta_task)
}

predict_sl <-  function(base_learners, sl, validation_task) {
  if(is.null(validation_task)) return(NULL)
  preds <- learner_predictions(base_learners, validation_task)
  meta_task <- sl3_Task$new(
    data = preds,
    covariates = names(base_learners),
    outcome = "Y"
  )
  sl$predict(meta_task)
}

evaluate_sl <- function(preds, validation_task, loss) {
  if(is.null(preds)) return(NULL)
  
  loss(preds, validation_task$Y)
  #loss_coverage(taus[2] - taus[1])(preds, validation_task$Y)
}

X <- byday %>%
  #filter(COD_ROR_EG %in% c("R1161050", "R1159199", "R1159447")) %>%
  filter(COD_ROR_EG %in% c("R1161050")) %>%
  group_by(COD_ROR_EG) %>%
  mutate(year = year,
         month = factor(month),
         wday = factor(wday),
         date = DAT_ENT,
         DAT_ENT = as.numeric(DAT_ENT),
         holiday = as.numeric(holiday),
         vacances_zone_c = as.numeric(vacances_zone_c),
         log_n = log(n),
         log_n_lag1 = lag(log_n, 1)) %>%
  select(date, DAT_ENT, log_n, log_n_lag1, year, month, wday, holiday, vacances_zone_c, inc, max_temp, min_temp) %>%
  filter(!is.na(log_n_lag1)) %>%
  as.data.frame() 

times <- tibble(
  DAT_ENT = unique(X$DAT_ENT)
) %>%
  mutate(time = 1:n())

X <- left_join(X, times)

ggplot(X, aes(x = date, y = log_n)) +
  geom_point() +
  facet_wrap(~COD_ROR_EG)

covars <- c("DAT_ENT", "year", "month", "wday", "holiday", "vacances_zone_c", "inc", "max_temp", "min_temp", "log_n_lag1")
#covars <- c("DAT_ENT", "year", "month", "wday")
task <- sl3_Task$new(
  data = X,
  covariates = covars,
  outcome = "log_n",
  folds = make_folds(
    n = nrow(X), 
    fold_fun = folds_rolling_origin_pooled, 
    t = nrow(times), 
    id = X$time, 
    time = X$time, 
    first_window = 365 * 2, 
    gap = 0, 
    validation_size = 1
  )
)

taus <- c(0.1, 0.9)
grf_learner <- Lrnr_grf$new(quantiles = taus, quantiles_pred = taus, num.threads = 1)
quantreg_learner <- Lrnr_quantreg$new(tau = taus)
qrnn_learner <- Lrnr_qrnn$new(tau = taus)

loss = loss_coverage(0.8)
#loss = loss_multi_quantile(taus)
#solnp <- Lrnr_solnp$new(learner_function = metalearner_linear_multivariate, eval_function = loss)
#solnp <- Lrnr_solnp$new(learner_function = metalearner_linear_multivariate, eval_function = loss_coverage(0.8))
solnp <- Lrnr_ga$new(learner_function = metalearner_linear_multivariate, eval_function = loss_coverage(0.8))

#
# Base learner predictions
#
plan(multisession, workers = 3)
folds <- task$folds %>% 
  map(function(fold) tibble(training_set = list(fold$training_set), validation_set = list(fold$validation_set))) %>%
  bind_rows() %>%
  mutate(
    training_task = map(training_set, make_task),
    validation_task = map(validation_set, make_task),
    base_learners = future_map(training_task, train_base_learners, .options = furrr_options(packages = c("tidyverse", "sl3")), .progress = TRUE)
  )

# Generate predictions
folds <- folds %>%
  mutate(
    performance = map2(base_learners, validation_task, evaluate_performance, loss = loss, .progress = "Evaluating base learners"),
    validation_predictions = map2(base_learners, validation_task, learner_predictions, .progress = "Generating predictions")
  )

folds$performance %>%
  bind_rows() %>%
  summarize_all(mean)

fold_learners <- folds %>%
  select(validation_predictions) %>%
  unnest(validation_predictions) %>%
  mutate(quantreg.lower = map_dbl(quantreg, function(x) unpack_predictions(x[[1]])[1]),
         quantreg.upper = map_dbl(quantreg, function(x) unpack_predictions(x[[1]])[2])) %>%
  mutate(grf.lower = map_dbl(grf, function(x) unpack_predictions(x[[1]])[1]),
         grf.upper = map_dbl(grf, function(x) unpack_predictions(x[[1]])[2]))
  #mutate(qrnn.lower = map_dbl(qrnn, function(x) unpack_predictions(x[[1]])[1]),
         #qrnn.upper = map_dbl(qrnn, function(x) unpack_predictions(x[[1]])[2]))
  
fold_learners %>%
  bind_cols(X[(nrow(X) - 363):nrow(X),]) %>%
  ggplot(aes(x = DAT_ENT, y = Y)) +
  geom_point() +
  geom_line(aes(y = quantreg.lower,  color = "quantreg"), alpha = 0.5) +
  geom_line(aes(y = grf.lower,  color = "grf"), alpha = 0.5) +
  #geom_line(aes(y = qrnn.lower,  color = "qrnn"), alpha = 0.5) +
  geom_line(aes(y = quantreg.upper,  color = "quantreg"), alpha = 0.5) +
  geom_line(aes(y = grf.upper,  color = "grf"), alpha = 0.5)
  #geom_line(aes(y = qrnn.upper,  color = "qrnn"), alpha = 0.5)

mean(fold_learners$Y >= fold_learners$quantreg.lower & fold_learners$Y <= fold_learners$quantreg.upper)
mean(fold_learners$Y >= fold_learners$grf.lower & fold_learners$Y <= fold_learners$grf.upper)
mean(fold_learners$Y >= fold_learners$qrnn.lower & fold_learners$Y <= fold_learners$qrnn.upper)


#
# Train SL
#
folds <- folds %>% mutate(
    sl_training_data = map(1:n(), function(i) bind_rows(folds$validation_predictions[1:i]), .progress = "SL training data"),
    sl_training_task = map2(base_learners, sl_training_data, make_sl_task, .progress = "SL training tasks")
  )

folds <- folds %>% mutate(
    sl = map(sl_training_task, make_sl, .progress = "Training SL"),
    sl_validation_task = lead(validation_task, 1),
    sl_prediction = pmap(list(base_learners, sl, sl_validation_task), predict_sl, .progress = "SL prediction"),
    sl_performance = pmap(list(sl_prediction, sl_validation_task), evaluate_sl, loss = loss, .progress = "SL performance")
  )

folds$performance %>%
  bind_rows() %>%
  summarize_all(mean)
mean(unlist(folds$sl_performance %>% map(mean, na.rm = TRUE)), na.rm = TRUE)

preds <- bind_cols(
  folds$sl_validation_task %>% map(function(task) tibble(task$X, Y = task$Y)) %>% bind_rows(),
  folds$sl_prediction %>% map(unpack_predictions) %>% map(as_tibble) %>% bind_rows()
)

mean(preds$V1 <= preds$Y & preds$V2 >= preds$Y)

preds %>%
  bind_cols(select(fold_learners[2:nrow(fold_learners),], -Y)) %>%
  ggplot(aes(DAT_ENT, exp(Y))) +
  #geom_line(aes(y = V1)) +
  geom_ribbon(aes(ymin = exp(V1), ymax = exp(V2)), alpha = 0.5) +
  geom_line(aes(y = exp(grf.lower))) +
  geom_line(aes(y = exp(grf.upper))) +
  geom_point()

last_preds <- bind_cols(
  X = folds$training_task[[364]]$X,
  Y = folds$training_task[[364]]$Y,
  bind_rows(map(folds$base_learners[[364]]$grf$predict(folds$training_task[[364]]), function(x) {
    tibble(
      lower = unpack_predictions(x[[1]])[1],
      upper = unpack_predictions(x[[1]])[2]
    )
  }))
)

mean(last_preds$lower <= last_preds$Y & last_preds$upper >= last_preds$Y)
ggplot(last_preds, aes(DAT_ENT, Y)) +
  geom_point() +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.5)

fold_learners %>%
  ggplot(aes(x = DAT_ENT, y = Y)) +
  geom_point() +
  geom_ribbon(aes(ymin = quantreg.lower, ymax = quantreg.upper, fill = "quantreg"), alpha = 0.5) +
  geom_ribbon(aes(ymin = grf.lower, ymax = grf.upper, fill = "grf"), alpha = 0.5) +
  geom_ribbon(aes(ymin = V1, ymax = V2, fill = "SL"), data = preds, alpha = 0.5) +
  facet_wrap(~COD_ROR_EG.R1161050)

loss = loss_coverage(0.8)
index <- 300
f <- Vectorize(function(x) { mean(loss(metalearner_linear_multivariate(c(x, 1 - x), folds$sl_training_task[[index]]$X), folds$sl_training_task[[index]]$Y)) })
curve(f(x), 0, 1)

folds$sl[[364]]$train(folds$sl_training_task[[364]])

solnp <- Lrnr_ga$new(learner_function = metalearner_linear_multivariate, eval_function = loss)
solnp$train(folds$sl_training_task[[364]])
