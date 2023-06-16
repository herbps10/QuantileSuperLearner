##' quantreg learner
##'
##' This learner uses \code{\link[quantreg]{quantreg}}
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
Lrnr_quantreg <- R6Class(
  classname = "Lrnr_quantreg", 
  inherit = Lrnr_base,
  portable = TRUE, 
  class = TRUE,
  public = list(
    initialize = function(tau = 0.5, ...) {
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
      
      args$x <- bind_cols(rep(1, nrow(task$X)), task$X)
      args$y <- outcome_type$format(task$Y)
      args$keep.inbag = TRUE

      fit_object <- Map(function(tau) quantreg::rq.fit(args$x, args$y, tau = tau), args$tau)
      
      return(fit_object)
    },

    # .predict takes a task and returns predictions from that task
    .predict = function(task = NULL) {
      mat <- matrix(unlist(Map(function(fit_object) {
        x <- as.matrix(bind_cols(intercept = rep(1, nrow(task$X)), task$X))
        predictions <- drop(x %*% fit_object$coefficients)
        return(predictions)
      }, self$fit_object)), ncol = length(self$fit_object), byrow = FALSE)
      if(length(self$params$tau) > 1) {
        pack_predictions(mat)
      }
      else {
        mat
      }
    },
    # list any packages required for your learner here.
    .required_packages = c("quantreg")
  )
)

