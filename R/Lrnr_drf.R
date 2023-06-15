##' quantreg learner
##'
##' This learner uses \code{\link[drf]{drf}}
##' to fit a distributional random forest.
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
Lrnr_drf <- R6Class(
  classname = "Lrnr_drf", 
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
      
      args$X <- as.data.frame(task$X)
      args$Y <- outcome_type$format(task$Y)
      
      fit_object <- sl3:::call_with_args(drf::drf, args)
      
      return(fit_object)
    },
    
    # .predict takes a task and returns predictions from that task
    .predict = function(task = NULL) {
      mat <- predict(private$.fit_object, newdata = data.frame(task$X), functional = "quantile", quantiles = self$params$tau)$quantile[,1,]
      if(length(self$params$tau) > 1) {
        pack_predictions(mat)
      }
      else {
        if(is.matrix(mat)) return(mat[, 1])
        mat
      }
    },
    # list any packages required for your learner here.
    .required_packages = c("drf")
  )
)
