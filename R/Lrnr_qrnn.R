##' qrnn learner
##'
##' This learner uses \code{\link[qrnn]{qrnn}}
##' to fit a quantile regression model.
##'
##' @docType class
##'
##' @importFrom R6 R6Class
##'
##' @export
##'
##' @keywords data
##'
##' @return A learner object inheriting from \code{\link{Lrnr_base}} with
##'  methods for training and prediction. For a full list of learner
##'  functionality, see the complete documentation of \code{\link{Lrnr_base}}.
##'
##' @format An \code{\link[R6]{R6Class}} object inheriting from
##'  \code{\link{Lrnr_base}}.
##'
##' @family Learners
##'
##' @section Parameters:
##'   - \code{tau="tau"}: quantile to fit.
##'   - \code{...}: Other parameters passed directly to 
##'      \code{\link[my_package]{quantreg}}. See its documentation for details. 
##'      Also, any additional parameters that can be considered by 
##'      \code{\link{Lrnr_base}}.
##' 
Lrnr_qrnn <- R6Class(
  classname = "Lrnr_qrnn", 
  inherit = Lrnr_base,
  portable = TRUE, 
  class = TRUE,
  public = list(
    initialize = function(tau = 0.5, n.hidden = 3, ...) {
      # this captures all parameters to initialize and saves them as self$params
      params <- args_to_list()
      super$initialize(params = params, ...)
    }
  ),
  private = list(
    # list properties your learner supports here.
    # Use sl3_list_properties() for a list of options
    .properties = c("continuous"),

    # .train takes task data and returns a fit object that can be used to generate predictions
    .train = function(task) {
      args <- self$params
      outcome_type <- self$get_outcome_type(task)
      
      args$x <- as.matrix(task$X)
      args$y <- as.matrix(task$Y)
      tau <- args$tau
      
      if(length(tau) == 1) {
        fit_object <- sl3:::call_with_args(qrnn::qrnn.fit, args, keep_all = TRUE)
      }
      else {
        fit_object <- sl3:::call_with_args(qrnn::mcqrnn.fit, args, keep_all = TRUE)
      }
      
      return(fit_object)
    },

    # .predict takes a task and returns predictions from that task
    .predict = function(task = NULL) {
      if(length(self$params$tau) == 1) {
        predictions <- qrnn::qrnn.predict(as.matrix(task$X), self$fit_object)
      }
      else {
        predictions <- pack_predictions(qrnn::mcqrnn.predict(as.matrix(task$X), self$fit_object))
      }
      
      return(predictions)
    },
    # list any packages required for your learner here.
    .required_packages = c("qrnn")
  )
)
