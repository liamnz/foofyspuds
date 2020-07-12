library(tidymodels)

# Data
data('credit_data', package = 'modeldata')

# Parameters
outcome <- 'Status'
major_class <- 'good'
splines <- 'Age'

# Make character to simulate how data might come in real life
data <-
  credit_data %>%
  as_tibble() %>%
  mutate(across(where(is.factor), as.character))

# Ensure response is a factor with the majority class in the first position.
# This is what `yardstick` expects for its calculations.
data[, outcome] <- forcats::fct_relevel(data[[outcome]], major_class)

# Split input data into training and test
data <- initial_split(data, strata = outcome)

# Compose pre-processing recipe. Define and prep non-tunable steps.
rec <-
  recipe(head(training(data), 5)) %>%
  update_role(everything(), new_role = 'predictor') %>%
  update_role(all_of(outcome), new_role = 'outcome') %>%
  add_role(has_type('nominal'), -all_of(outcome), new_role = 'nominal_predictor') %>%
  step_naomit(all_of(outcome)) %>%
  step_nzv(all_predictors()) %>%
  step_unknown(has_role('nominal_predictor'), new_level = '[Unknown]') %>%
  step_other(has_role('nominal_predictor'), other = '[Pooled]') %>%
  step_medianimpute(all_numeric()) %>%
  prep(training = training(data), verbose = TRUE)

# Define tunable steps
if (!is.null(splines)){
  rec <- rec %>% step_ns(all_of(splines), deg_free = tune())
}







