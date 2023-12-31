cv_risks <- setup %>% 
  mutate(cv_risk = map(cv_fit, `[[`, "cv_risk")) %>%
  select(name, quantile, cv_risk) %>%
  unnest(cv_risk) %>%
  mutate(learner = str_replace_all(learner, "Lrnr_", ""),
         learner = str_replace_all(learner, "_.+$", ""),
         learner = str_replace_all(learner, "SuperLearner", "Quantile\nSuperLearner"))

cv_risks %>%
  ggplot(aes(x = risk, y = reorder(learner, risk))) +
  geom_point() +
  facet_wrap(~quantile, scales = "free", ncol = 1) +
  labs(x = "Cross-validated risk", y = "Estimator")

ggsave("median_quantiles.png", width = 8, height = 4)

cv_risks %>%
  ggplot(aes(x = risk, y = learner, color = factor(quantile))) +
  geom_point() +
  facet_wrap(~quantile + name, scales = "free")

cv_risks %>%
  ggplot(aes(x = risk, y = reorder(learner, risk))) +
  geom_point() +
  facet_grid(quantile~name, scales = "free_x")

cv_risks %>%
  ggplot(aes(x = risk, y = quantile, color = learner)) +
  geom_point() +
  facet_wrap(~name, scales = "free_x")

preds <- tibble(dat, pred0.1 = sl_fit_0.1$predict(), pred0.9 = sl_fit_0.9$predict())
ggplot(preds, aes(x = x, y = y)) +
  geom_point(size = 0.1) +
  geom_ribbon(aes(ymin = pred0.1, ymax = pred0.9), alpha = 0.5)

preds %>% mutate(within = pred0.1 <= y & pred0.9 >= y) %>% 
  summarize(mean = mean(within))

preds <- as_tibble(MASS::Boston) %>%
  mutate(pred0.1 = sl_fit_0.1$predict(),
         pred0.9 = sl_fit_0.9$predict())

preds %>% mutate(within = pred0.1 <= medv & pred0.9 >= medv) %>%
  summarize(mean = mean(within))
