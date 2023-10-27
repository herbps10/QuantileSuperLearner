library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(furrr)
library(progressr)

download_data <- function(id, tz) {
  print(id)
  data_target <- read_csv(glue::glue("https://raw.githubusercontent.com/wentingwang94/probabilistic-solar-forecasting/main/data/ENS_{id}.csv")) %>%
    mutate(time_utc = lubridate::parse_date_time(UTC, orders = c("%Y/%m/%d %H:%M"), tz = "UTC"),
           time = lubridate::force_tzs(time_utc, tzone = "UTC", tzone_out = tz)) %>%
    mutate(lagged = lag(NSRDB_GHI, 1),
           lagged2 = lag(NSRDB_GHI, 2),
           direction = lagged - lagged2) %>%
    filter(`Solar Zenith Angle` < 85) %>%
    mutate(month = lubridate::month(time),
           zenith = `Solar Zenith Angle`,
           yday = lubridate::yday(time),
           lagged_oneday = lag(NSRDB_GHI, 1)
    ) %>%
    filter(!is.na(lagged), !is.na(lagged_oneday))
}

data <- tibble(id = c("BON", "DRA", "FPK", "GWN", "PSU", "SXF", "TBL"),
               tz = c("US/Central", "US/Pacific", "US/Mountain", "US/Central", "US/Eastern", "US/Central", "US/Mountain")) %>%
  mutate(data = map2(id, tz, download_data),
         data = map(data, function(df) filter(df, lubridate::hour(time) == 12)))

fit_quantile <- function(data, tau, p = NULL) {
  if(!is.null(p)) p()
  ensemble_members <- paste0("EC_GHI_", 1:50)
  test_times <- data %>% filter(lubridate::year(time) >= 2020) %>% pull(time)
  test_indices <- which(data$time %in% test_times)

  preds <- matrix(nrow = nrow(data), ncol = 6)
  for(index in test_indices) {
    print(index)
    form1 <- paste("NSRDB_GHI ~ ", paste0(ensemble_members, collapse = " + "))
    form2 <- paste0(form1, " + zenith + lagged_oneday")
    form_qgam <- paste0(form1, " + s(zenith) + s(lagged_oneday)")
    
    x <- as.matrix(data[1:(index - 1), c(ensemble_members, "zenith", "lagged_oneday" )])
    newx <- as.matrix(data[index, c(ensemble_members, "zenith", "lagged_oneday")])
    y <- data[1:(index - 1),][["NSRDB_GHI"]]
    
    fit1 <- quantreg::rq(form1, data = data[1:(index - 1),], tau = tau)
    fit2 <- quantreg::rq(form2, data = data[1:(index - 1),], tau = tau)
    fit3 <- lightgbm::lightgbm(x, y, params = list(objective = "quantile", alpha = tau), nrounds = 5e2, verbose = -1)
    fit4 <- grf::quantile_forest(x, y)
    fit5 <- qgam::qgam(as.formula(form_qgam), data = data[1:(index - 1),], qu = tau)
    fit6 <- qrnn::qrnn.fit(x, as.matrix(y, ncol = 1), n.hidden = 2, tau = tau)
    
    preds[index, 1] <- predict(fit1, newdata = data[index, ])
    preds[index, 2] <- predict(fit2, newdata = data[index, ])
    preds[index, 3] <- predict(fit3, data = newx)
    preds[index, 4] <- predict(fit4, newdata = newx, quantiles = tau)$predictions[,1]
    preds[index, 5] <- predict(fit5, newdata = data[index, ], quantiles = tau)
    preds[index, 6]  <- qrnn::qrnn.predict(x = newx, parms = fit6)
  }
  
  list(preds = preds, test_indices = test_indices)
}

apply_aggregation <- function(data, preds, method, quantile, p = NULL) {
  if(!is.null(p)) p()
  loss.type <- list(name = "pinball", tau = quantile)
  opera::mixture(data[preds$test_indices, ][["NSRDB_GHI"]], preds$preds[preds$test_indices, ], model = method, loss.type = loss.type)
}

plan(multisession, workers = 16)

with_progress({
  print("Predictions...")
  p <- progressr::progressor(steps = nrow(data) * 3)
  preds <- expand_grid(data, quantile = c(0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975)) %>%
    mutate(preds = future_map2(data, quantile, fit_quantile, p = p))
  
  print("Ensemble...")
  p <- progressr::progressor(steps = nrow(data) * 3)
  fits <- expand_grid(preds, tibble(method = c("SuperLearner", "EWA", "BOA"))) %>%
    mutate(fit = future_pmap(list(data, preds, method, quantile), apply_aggregation, p))
})

write_rds(fits, "solar_results.rds")
