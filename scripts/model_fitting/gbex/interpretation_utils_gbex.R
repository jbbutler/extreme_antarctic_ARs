# Function taken from the gbex source code in GitHub
# https://github.com/JVelthoen/gbex
# Rewriting the functions here because the package by default only computes PDPs using the training
# data, but we want to use both the training data and the testing data

compute_quantile_PDP_1D <- function(object, tau, X_data, var_name, grid_points=50, extents=NULL){

    if (is.null(extents)) {
        values = seq(min(X_data[[var_name]]), max(X_data[[var_name]]), length.out=grid_points)
    } else {
        values = seq(extents[1], extents[2], length.out=grid_points)
    }    

  quant_PD = numeric(length(values))
  for(icnt in 1:length(values)){
    value = values[icnt]
    # copy the dataframe to replace a single value
    df_value = data.frame(X_data)
    df_value[var_name] = value
    quant = predict(object, df_value, what="quant", probs=tau)
    quant_PD[icnt] = mean(quant)
  }

  quant_PD = data.frame(Q = quant_PD)
  quant_PD[var_name] = values
  
  output = list(PD_df = quant_PD,
              tau = tau)
  return(output)
}

compute_quantile_PDP_2D <- function(object, tau, X_data, var_names, grid_points=50, extents=NULL) {

    if(is.null(extents)){
        values = expand.grid(seq(min(X_data[[var_names[1]]]), max(X_data[[var_names[[1]]]]), length.out=grid_points),
                           seq(min(X_data[[var_names[2]]]), max(X_data[[var_names[[2]]]]), length.out=grid_points))
        colnames(values) = var_names
      } else{
        values = expand.grid(seq(extents[1], extents[2], length.out=grid_points),
                           seq(extents[3], extents[4], length.out=grid_points))
        colnames(values) = var_names
    }

    quant_PD = numeric(nrow(values))
    for(icnt in 1:nrow(values)){
      df_value = data.frame(X_data)
      df_value[var_names] = values[icnt,]
      quant = predict(object, df_value, what="quant", probs=tau)
      quant_PD[icnt] = mean(quant)
    }

    quant_PD = data.frame(Q = quant_PD)
    quant_PD = cbind(quant_PD,values)

    output = list(PD_df = quant_PD,
              tau = tau)
  return(output)
}

permutation_score <- function(object, X_data, y_data, n_reps=500, verbose=FALSE) {

    baseline_deviances <- dev_per_step(object, y=y_data, X=X_data)
    baseline_pred <- baseline_deviances[length(baseline_deviances)]
    
    cols <- colnames(X_data)
    avg_perm_scores <- rep(NA, length(cols))
    
    for (i in 1:length(cols)) {
        feature <- cols[i]
        permuted_devs <- rep(NA, n_reps)
        for (j in 1:n_reps) {
            permuted_data <- data.frame(X_data)
            permuted_data[[feature]] <- permuted_data[[feature]][sample(1:nrow(permuted_data))]
            deviances_per_step <- dev_per_step(object, y=y_data, X=permuted_data)
            permuted_devs[j] <- deviances_per_step[length(deviances_per_step)]
        }
        
        avg_perm_scores[i] <- mean(permuted_devs) - baseline_pred
        if (verbose) {
            print(paste0('Finishing ', feature))
        }
    }

    return(data.frame(features=cols, avg_perm_diff=avg_perm_scores))
}

