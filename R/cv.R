my_cv_sl <- function(lrnr_sl, task, eval_fun) {
  # check arguments
  if (!inherits(lrnr_sl, "Lrnr_sl")) {
    stop("lrnr_sl must be a Lrnr_sl object")
  }
  # cross-validate the SL
  cv_sl_lrnr <- make_learner(Lrnr_cv, lrnr_sl, full_fit = TRUE)
  print("Starting training")
  cv_sl_fit <- cv_sl_lrnr$train(task)
  print("Finished training")
  full_fit <- cv_sl_fit$fit_object$full_fit
  # TODO: extract loss function from lrnr_sl where possible
  full_risk <- full_fit$cv_risk(eval_fun)
  sl_risk <- cv_sl_fit$cv_risk(eval_fun)
  # replace revere cv sl risk with nested cv sl risk as needed
  stack_risks <- full_risk[full_risk$learner != "SuperLearner"]
  set(sl_risk, , "learner", "SuperLearner")
  risks <- rbind(stack_risks, sl_risk)
  
  # remove full-fit SL coefficients from the table
  risks <- risks[, -"coefficients", with = FALSE]
  
  
  # gather the fold-specific SL coefficients
  if (!is.null(names(full_fit$coefficients))) {
    coefs <- do.call(
      rbind,
      lapply(seq_along(cv_sl_fit$fit_object$fold_fits), function(i) {
        coefs <- stats::coef(cv_sl_fit$fit_object$fold_fits[[i]]$fit_object$cv_meta_fit)
        c("fold" = i, coefs)
      })
    )
  } else {
    coefs <- lapply(seq_along(cv_sl_fit$fit_object$fold_fits), function(i) {
      cv_sl_fit$fit_object$fold_fits[[i]]$fit_object$cv_meta_fit
    })
  }

  # Get predictions from each candidate learner
  candidate_preds <- matrix(ncol = length(cv_sl_fit$fit_object$full_fit$learner_fits), nrow = nrow(task$X))
  colnames(candidate_preds) <- names(cv_sl_fit$fit_object$full_fit$learner_fits)

  for(index in seq_along(cv_sl_fit$fit_object$folds)) {
	  validation_set <- cv_sl_fit$fit_object$folds[[index]]$validation
    validation_task <- task$subset_task(validation_set)

    for(learner in colnames(candidate_preds)) {
      candidate_preds[validation_set, learner] <- cv_sl_fit$fit_object$fold_fits[[index]]$learner_fits[[learner]]$predict(validation_task)   
    }

  }

  print("Cross-validated risk:")
  print(risks)

  #return_obj <- list("coef" = coefs, "cv_risk" = risks, cv_sl_fit = cv_sl_fit)
  return_obj <- list("coef" = coefs, "cv_risk" = risks, preds = cv_sl_fit$predict(), candidate_preds = candidate_preds)
  return(return_obj)
}
