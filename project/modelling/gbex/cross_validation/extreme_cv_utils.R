# Utils functions for carrying out the gbex model fitting.
#
# Jimmy Butler
# 11/2025

gpd_net_deviance <- function(z, sigma, xi) {
    # z is the exceedances of y over some threshold u
    # sigma is the candidate GPD scale parameters for each obs
    # xi is the candidate GPD shape parameters for each obs
  
  n <- length(z)
  log_like_parts <- numeric(n)
  
  if (sum(idx_xi_zero) > 0) {
    y_zero <- y[idx_xi_zero]; sigma_zero <- sigma[idx_xi_zero]
    log_like_parts[idx_xi_zero] <- -log(sigma_zero) - y_zero / sigma_zero
  }
  
  if (sum(idx_xi_nonzero) > 0) {
    y_non <- y[idx_xi_nonzero]; sigma_non <- sigma[idx_xi_nonzero]; xi_non <- xi[idx_xi_nonzero]
    term <- 1 + xi_non * y_non / sigma_non
    valid_term <- term > 0
    
    log_like_parts[idx_xi_nonzero][valid_term] <- -log(sigma_non[valid_term]) -
      (1 / xi_non[valid_term] + 1) * log(term[valid_term])
    log_like_parts[idx_xi_nonzero][!valid_term] <- -Inf
  }
  
  total_log_like <- sum(log_like_parts[is.finite(log_like_parts)])
  return(-total_log_like)
}