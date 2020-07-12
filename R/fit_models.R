#' Cross-validate Model Performance
#'
#' @import parsnip
#'
#' @export
cross_val_auc <- function(model_spec, x, y, folds){
  k <- max(folds)
  auc <- double(k)
  for (i in 1:k){
    mdl_fit <- fit_xy(model_spec, x[folds != i, ], y[folds != i])
    mdl_est <- predict(mdl_fit, x[folds == i, ], type = "prob")
    auc[i] <- yardstick::roc_auc_vec(y[folds == i], mdl_est[[1]], options = list(quiet = TRUE))
  }
  mean(auc)
}

#' Perform a grid-search for the best hyper-parameters
#'
#' @import future
#' @import parsnip
#'
#' @export
hyper_search <- function(candidates, x, y, folds, parallel){

  k <- max(folds)

  if (parallel){

    auc_future <- list()
    plan(multiprocess)
    for(i in seq_along(candidates)){
      auc_future[[i]] <- future({cross_val_auc(candidates[[i]], x, y, folds)},
                                packages = "parsnip")
    }
    auc_cv <- unlist(values(auc_future))

  } else {

    auc_cv <- double(length(candidates))
    for(i in seq_along(candidates)){
      auc_cv[i] <- cross_val_auc(candidates[[i]], x, y, folds)
    }
  }

  fit_xy(candidates[[which.max(auc_cv)]], x, y)
}

#' Fit a regularised logistic regression with 'glmnet'
#'
#' @export
fit_glmnet <- function(x, y, folds, parallel = TRUE){

  if (parallel){
    n_cores <- parallel::detectCores()
    workers <- parallel::makeCluster(n_cores, type = "SOCK")
    doParallel::registerDoParallel(workers)
  }

  glmnet_cv <- glmnet::cv.glmnet(x, y, family = "binomial", parallel = parallel, foldid = folds)

  if (parallel){
    parallel::stopCluster(workers)
  }

  glmnet_cv$glmnet.fit
}

#' Fit a decision tree with 'rpart'
#'
#' @import parsnip
#' @import dials
#'
#' @export
fit_rpart <- function(x, y, folds, parallel = TRUE){
  rpart_spec <- decision_tree(
      mode = "classification",
      tree_depth = varying(),
      min_n = varying()
      ) %>%
    set_engine("rpart")

  rpart_hypergrid <-
    grid_regular(
      tree_depth %>% range_set(c(2, 30)),
      min_n      %>% range_set(c(10, 100)),
      levels = 4
    )

  rpart_candidates <- merge(rpart_spec, rpart_hypergrid)

  hyper_search(rpart_candidates, x, y, folds, parallel = parallel)
}

#' Fit a random forest with 'ranger'
#'
#' @import parsnip
#' @import dials
#'
#' @export
fit_ranger <- function(x, y, folds, seed, parallel = TRUE){

  p <- ncol(x)
  mtry_sqrt <- floor(sqrt(p))
  mtry_log  <- max(1, floor(log(p)))

  if (mtry_log == mtry_sqrt){
    mtry_range <- range_set(mtry, c(1, 2))
  } else {
    mtry_range <- range_set(mtry, c(mtry_log, mtry_sqrt))
  }

  ranger_spec <-
    rand_forest(
      "classification",
      mtry = varying(),
      min_n = varying(),
      trees = 500
    ) %>%
    set_engine("ranger", seed = !!seed)

  ranger_hypergrid <- grid_regular(
    mtry_range,
    min_n %>% range_set(c(10, 100)),
    levels = 4
  )

  ranger_candidates <- merge(ranger_spec, ranger_hypergrid)

  hyper_search(ranger_candidates, x, y, folds, parallel = parallel)
}

#' Fit a multilayer perceptron with with 'nnet'
#'
#' @import parsnip
#' @import dials
#'
#' @export
fit_nnet <- function(x, y, folds, parallel = TRUE){

  nnet_spec <- mlp(mode = "classification",
                   penalty = varying(),
                   hidden_units = varying(),
                   epochs = 300) %>%
    set_engine("nnet")

  hidden_max <- floor(nrow(x) / (5 * (ncol(x) + 1)))

  nnet_hypergrid <- grid_regular(
    hidden_units %>% range_set(c(2, hidden_max)),
    penalty,
    levels = 4
  )

  # stack the larger architectures at the front so that these enter the parallel
  # processing first
  nnet_candidates <- rev(merge(nnet_spec, nnet_hypergrid))

  hyper_search(nnet_candidates, x, y, folds, parallel = parallel)
}
