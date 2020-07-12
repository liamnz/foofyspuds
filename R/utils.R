#' Internal Utility Functions
#'
#'
#'@export
make_dummies <- function(rec){
    prep(step_dummy(rec, all_nominal(), -all_outcomes(), one_hot = TRUE))
}
