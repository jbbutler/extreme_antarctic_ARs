# Helper functions for carrying out k-fold CV for intermediate conditional quantile modelling.
#
# Jimmy Butler
# November 17, 2025

pinball_loss <- function(y_true, q_pred, tau) {
  err <- y_true - q_pred
  loss <- mean(ifelse(err > 0, tau * err, (tau - 1) * err))
  return(loss)
}