#' Get a Dataset Ready For Modelling
#'
#' @import recipes
#' @importFrom rsample initial_split training testing
#'
#' @export
prepare_data <- function(data, response, target, verbose = TRUE){

  if (nrow(dplyr::distinct(tidyr::drop_na(input[, response]))) != 2){
    stop("Response variable must have exactly two levels")
  }

  if (verbose){
    start_time <- proc.time()[3]
    cat("Pre-processing data, please wait...\n", sep = "")
  }

  # Ensure response is a factor with the target class in the last position. This
  # is what `yardstick` expects for its calculations.
  input[, response] <- forcats::fct_relevel(input[[response]], target, after = Inf)

  data <- initial_split(data, strata = {{response}})

  rec <- recipe(head(training(data), 5)) %>%
    update_role(-{{response}}, new_role = "predictor") %>%
    update_role( {{response}}, new_role = "outcome") %>%
    step_naomit({{response}}) %>%
    step_nzv(all_predictors()) %>%
    step_modeimpute(all_nominal(), -{{response}}) %>%
    step_medianimpute(all_numeric()) %>%
    step_other(all_nominal(), -{{response}}, other = "(Pooled)")

  steps <- prep(rec, training(data), retain = FALSE)
  train <- bake(steps, training(data))
  test  <- bake(steps, testing(data))

  if (verbose){
    cat("Pre-processing complete, duration: ", proc.time()[3] - start_time, "s\n\n", sep = "")
    print(steps)
  }

  list(recipe = steps, train  = train, test = test)
}
