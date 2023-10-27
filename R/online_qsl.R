library(opera)

data <- read_csv("data/Tetuan City power consumption.csv") %>%
  mutate(DateTime = lubridate::parse_date_time(DateTime, orders = "m/d/Y H:M"))

analysis_data <- data %>%
  mutate(date = lubridate::as_date(DateTime),
         total_power = `Zone 1 Power Consumption` + `Zone 2  Power Consumption` + `Zone 3  Power Consumption`) %>%
  group_by(date) %>%
  summarize(
    mean_power = mean(total_power), 
    mean_temperature = mean(Temperature), 
    mean_humidity = mean(Humidity), 
    mean_wind_speed = mean(`Wind Speed`)
  ) %>%
  arrange(date) %>%
  mutate(mean_power_lag = lag(mean_power, 1), wday = factor(lubridate::wday(date))) %>%
  filter(!is.na(mean_power_lag))

ggplot(analysis_data, aes(x = date, y = mean_power)) +
  geom_point()

preds <- matrix(nrow = nrow(analysis_data), ncol = 5)
f <- mean_power ~ mean_temperature + mean_humidity + mean_wind_speed + mean_power_lag
f_qgam <- mean_power ~ s(mean_temperature) + s(mean_humidity) + s(mean_wind_speed) + s(mean_power_lag)
for(t in 61:nrow(analysis_data)) {
  print(t)
  x <- model.matrix(f, analysis_data[1:(t - 1), ])
  y <- analysis_data[1:(t - 1),]$mean_power
  fit1 <- quantreg::rq(f, data = analysis_data[1:(t - 1),], tau = 0.5)
  fit2 <- grf::quantile_forest(x, y, quantiles = 0.5, num.trees = 4e3)
  fit3 <- lightgbm::lightgbm(x, label = y, params = list(objective = "quantile", alpha = 0.5), nrounds = 2e3, verbose = -1)
  fit4 <- qgam::qgam(f_qgam, data = analysis_data[1:(t - 1), ], qu = 0.5)
  fit5 <- qrnn::qrnn.fit(x, y, tau = 0.5)
  
  preds[t, 1] <- predict(fit1, newdata = analysis_data[t,])
  preds[t, 2] <- predict(fit2, newdata = as.matrix(model.matrix(f, analysis_data[t, ])))$predictions
  preds[t, 3] <- predict(fit3, data = model.matrix(f, analysis_data[t,]))
  preds[t, 4] <- predict(fit4, newdata = analysis_data[t,])
  preds[t, 5] <- predict(x = analysis_data[t,], parms = fit5)
}
colnames(preds) <- c("quantreg", "grf", "lightgbm", "qgam")

matplot(preds[index, ], type = "l")
points(analysis_data$mean_power[index])

index <- 61:nrow(analysis_data)

mean(opera::loss(preds[index,1], analysis_data$mean_power[index], loss.type = list(name = "pinball", tau = 0.5)))
mean(opera::loss(preds[index,2], analysis_data$mean_power[index], loss.type = list(name = "pinball", tau = 0.5)))
mean(opera::loss(preds[index,3], analysis_data$mean_power[index], loss.type = list(name = "pinball", tau = 0.5)))
mean(opera::loss(preds[index,4], analysis_data$mean_power[index], loss.type = list(name = "pinball", tau = 0.5)))

mix_ewa <- opera::mixture(Y = analysis_data$mean_power[index], experts = preds[index,], model = "EWA", loss.type = list(name = "pinball", tau = 0.5))
mix_boa <- opera::mixture(Y = analysis_data$mean_power[index], experts = preds[index,], model = "BOA", loss.type = list(name = "pinball", tau = 0.5))
mix_sl  <- opera::mixture(Y = analysis_data$mean_power[index], experts = preds[index,], model = "SuperLearner", loss.type = list(name = "pinball", tau = 0.5))

plot(mix_boa)
plot(mix_ewa)
  
mix_ewa$loss
mix_boa$loss
mix_sl$loss

plot(mix_boa)
plot(mix_ewa)
plot(mix_sl)
