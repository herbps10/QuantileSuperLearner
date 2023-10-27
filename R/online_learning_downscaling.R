library(tidyverse)
library(lubridate)

observed_data <- read_csv("data/ithaca_noaa.csv") %>%
  rename(date = DATE, observed_tmax = TMAX) %>%
  select(date, observed_tmax)

ithaca <- AOI::geocode(location = "Ithaca, NY", pt = TRUE)
gmet_data <- getGridMET(ithaca, c("tmmx", "pr", "sph", "vpd", "vs"), startDate = "1979-01-01", endDate = "2022-12-31")
gmet_data <- gmet_data %>%
  mutate(f = weathermetrics::kelvin.to.fahrenheit(tmmx),
         year = lubridate::year(date),
         yday = lubridate::yday(date),
         month = factor(lubridate::month(date)),
         date = as.Date(date))

combined_data <- gmet_data %>%
  left_join(observed_data, by = "date") %>%
  filter(!is.na(observed_tmax))

gmet_data %>%
  ggplot(aes(x = sph, y = tmmx, color = pr)) +
  geom_point()

covars <- c("f", "yday", "f", "pr", "sph", "vpd", "vs")


preds <- matrix(nrow = nrow(combined_data), ncol = 4)
for(y in 2000:2022) {
  print(y)
  training_data <- combined_data %>% filter(year < y)
  test_data <- combined_data %>% filter(year == y)
  
  X <- combined_data[,covars]
  Y <- combined_data$observed_tmax
  
  fit_qgam <- qgam::qgam(observed_tmax ~ s(f) + s(yday) + s(pr) + s(sph) + s(vpd) + s(vs), data = training_data, qu = 0.99)
  fit_qreg <- quantreg::rq(observed_tmax ~ f + month + f * month + pr + sph + vpd + vs, data = training_data, tau = 0.99)
  fit_grf <- grf::quantile_forest(training_data[,covars], training_data$observed_tmax, num.trees = 2e3, quantiles = 0.99)
  fit_gbm <- lightgbm::lightgbm(as.matrix(training_data[,covars]), label = training_data$observed_tmax, params = list(objective = "quantile", alpha = 0.5), nrounds = 2e3, verbose = -1)
  
  preds[combined_data$year == y, 1] <- predict(fit_qreg, newdata = test_data)
  preds[combined_data$year == y, 2] <- predict(fit_qgam, newdata = test_data)
  preds[combined_data$year == y, 3] <- predict(fit_grf, newdata = test_data[,covars])$predictions
  preds[combined_data$year == y, 4] <- predict(fit_gbm, data = as.matrix(test_data[,covars]))
}

colnames(preds) <- c("qreg", "qgam", "grf", "lightgbm")

indices <- !is.na(preds[,1])

mix_ewa <- opera::mixture(Y = combined_data$observed_tmax[indices], experts = preds[indices, ], method = "EWA")
mix_boa <- opera::mixture(Y = combined_data$observed_tmax[indices], experts = preds[indices, ], method = "BOA")
mix_sl  <- opera::mixture(Y = combined_data$observed_tmax[indices], experts = preds[indices, ], method = "SuperLearner")

plot(mix_ewa)
plot(mix_boa)
plot(mix_sl)

pinball <- function(x, y, tau = 0.5) {
  opera::loss(x, y, loss.type = list(name = "pinball", tau = tau))
}

mean(pinball(predict(fit), combined_data$observed_tmax, tau = 0.99))
mean(pinball(predict(fit_qreg), combined_data$observed_tmax, tau = 0.99))
mean(pinball(predict(fit_grf)$predictions, combined_data$observed_tmax, tau = 0.99))

ggplot(combined_data, aes(x = f, y = observed_tmax)) +
  geom_point()

plot(fit)

ggplot(combined_data, aes(x = tmax_hat, y = observed_tmax)) +
  geom_point() +
  geom_abline(slope = 1, lty = 2)

data <- climateR::getMACA(ithaca, c("tasmax", "tasmin", "rhsmin", "rhsmax", "pr"), timeRes = "day", startDate = "1979-01-01", endDate = "2030-12-31")
future_data <- climateR::getMACA(ithaca, "tasmax", timeRes = "day", startDate = "2006-01-01", endDate = "2100-12-31", scenario = "rcp85")
future_data <- future_data %>%
  mutate(modeled = weathermetrics::kelvin.to.fahrenheit(tasmax_CCSM4_r6i1p1_rcp85),
         yday = lubridate::yday(date))

ggplot(future_data, aes(x = date, y = tasmax_CCSM4_r6i1p1_rcp85)) +
  geom_point() +
  geom_smooth()

analysis_data <- data %>%
  mutate(date = as.Date(date)) %>%
  left_join(observed_data) %>%
  mutate(modeled = weathermetrics::kelvin.to.fahrenheit(tasmax_CCSM4_r6i1p1_historical)) %>%
  select(date, tasmax_CCSM4_r6i1p1_historical, pr_CCSM4_r6i1p1_historical, modeled, observed_tmax, tasmin_CCSM4_r6i1p1_historical) %>%
  mutate(yday = lubridate::yday(date))

analysis_data <- analysis_data[complete.cases(analysis_data),]

plot(analysis_data$modeled, analysis_data$observed_tmax)

df <- model.matrix(observed_tmax ~ -1 + modeled, data = analysis_data)

model  <- quantreg::rq(observed_tmax ~ factor(lubridate::month(date)) + modeled, data = analysis_data, tau = 0.5)
model2 <- quantreg::rq(observed_tmax ~ factor(lubridate::month(date)) + modeled + pr_CCSM4_r6i1p1_historical + tasmin_CCSM4_r6i1p1_historical, data = analysis_data, tau = 0.5)

mean(abs(analysis_data$observed_tmax - predict(model)))
mean(abs(analysis_data$observed_tmax - predict(model2)))

mean(opera::loss(analysis_data$observed_tmax, predict(model), loss.type = list(name = "pinball", tau = 0.95)))
mean(opera::loss(analysis_data$observed_tmax, predict(model2), loss.type = list(name = "pinball", tau = 0.95)))

future_data$yhat <- predict(model, newdata = future_data)
future_data$yhat2 <- predict(model2, newdata = future_data)
future_data$yhat3 <- predict(model3, newdata = model.matrix(~ -1 + modeled, data = future_data))$predictions[,1]

ggplot(aes(x = date, y = yhat), data = future_data) +
  geom_point()

future_data %>%
  group_by(year = year(date)) %>%
  summarize(y = max(yhat3)) %>%
  ggplot(aes(x = year, y = y)) +
  geom_point()
