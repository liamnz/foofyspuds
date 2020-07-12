#' Get a Dataset Ready For Modelling
#'
#' @import recipes
#' @importFrom rsample initial_split training testing
#'
#' @export
prepare_data <- function(data, response, target, verbose = TRUE){

  # Ensure response is a factor with the target class in the first position.
  # This is what `yardstick` expects for its calculations.
  data[, response] <- forcats::fct_relevel(data[[response]], target)

  if (nlevels(data[[response]] != 2)){
    stop("Response variable must exactly two levels")
  }

  if (verbose){
    start_time <- proc.time()[3]
    cat("Pre-processing training data, please wait...\n\n", sep = "")
  }

  data <- initial_split(data, strata = {{response}})

  rec <- recipe(head(training(data), 5)) %>%
    update_role(-{{response}}, new_role = "predictor") %>%
    update_role( {{response}}, new_role = "outcome") %>%
    step_naomit({{response}}) %>%
    step_nzv(all_predictors()) %>%
    step_modeimpute(all_nominal(), -{{response}}) %>%
    step_medianimpute(all_numeric()) %>%
    step_other(all_nominal(), -{{response}}, other = "minor categories") %>%
    prep(training(data), verbose = verbose)

  if (verbose){
    cat("Training data processed, duration: ", proc.time()[3] - start_time, "s\n\n", sep = "")
    print(rec)
  }

  if (verbose){
    start_time <- proc.time()[3]
    cat("\nApplying pre-processing to test data, please wait...\n", sep = "")
  }

  test <- bake(rec, testing(data))

  if (verbose){
    cat("Test data processed, duration: ", proc.time()[3] - start_time, "s\n\n", sep = "")
  }

  list(recipe = rec, test_data = test)
}
