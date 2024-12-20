
#	est<- rq(fmla.black, tau=.01, method="pfn")
#	summary.rq.extreme(est, subsample.fraction=4000/32000, R=100) 
#   	summary.rq(est, se="ker", mofn=2000)
#	summary.rq(est, se="boot", mofn=2000) 
#	summary.rq(est, se="boot", mofn=2000) 
#	summary.rq(est, se="ker”)


# This is an R-function to give summary statistics of Extremal QR

summary.rq.extreme<- function (object, subsample.fraction=.2, R=100, method = "fn", alpha = .10, spacing = 20, ...) 

{
    mt 		<- terms(object)
    m 		<- model.frame(object)
    y 		<- model.response(m)
    x 		<- model.matrix(mt, m, contrasts = object$contrasts)
    wt 		<- model.weights(m)
    tau 	<- object$tau
    eps 	<- .Machine$double.eps^(2/3)
    coef 	<- coefficients(object)
    if (is.matrix(coef))  
	coef 	<- coef[, 1]
    vnames 	<- dimnames(x)[[2]]
    resid 	<- object$residuals
    n 		<- length(resid)
    p	 	<- length(coef)
    rdf 	<- n - p
    if (!is.null(wt)) {
       	resid 	<- resid * wt
       	x 	<- x * wt
       	y 	<- y * wt
   	}

   infer.object 	<- rq.infer.extreme(x, y, tau = tau, subsample.fraction=subsample.fraction, spacing = spacing, R=R, method = method, ...)

    coef 		<- array(coef, c(p, 5))
    dimnames(coef) 	<- list(vnames,  c("Value", "Psedo Std. Er.", "Lower  5% Bound",  "Upper 95% Bound", "Bias-Corrected Estimate") )
    coef[, 3] 		<- infer.object$ciL.e		
    coef[, 4] 		<- infer.object$ciU.e		      
    coef[, 2] 		<- (coef[, 4]- coef[, 3])/(2*qnorm(1-alpha/2)) 
    coef[, 5] 		<- infer.object$BC.betat
    object 		<- object[c("call", "terms")]
    object$R		<-  infer.object$R
    object$coefficients	<- coef
    object$tau 		<- tau
    class(object) 	<- "summary.rq"
    object
}



# function that does not complain about singular designs and records instead a NaN

rq.fit.fn.nostop<- function (x, y, tau = 0.5, beta = 0.99995, eps = 1e-06) 
{
    n 		<- length(y)
    p 		<- ncol(x)
    if (n != nrow(x))  
	stop("x and y don't match n")
    if (tau < eps || tau > 1 - eps) 
        stop("tau is outside (0,1)")
    rhs 	<- (1 - tau) * apply(x, 2, sum)
    d 		<- rep(1, n)
    u 		<- rep(1, n)
    wn 		<- rep(0, 10 * n)
    wn[1:n] 	<- (1 - tau)
    z 		<- .Fortran("rqfn", as.integer(n), as.integer(p), a = as.double(t(as.matrix(x))), 
        		c = as.double(-y), rhs = as.double(rhs), d = as.double(d), 
        		as.double(u), beta = as.double(beta), eps = as.double(eps), 
        		wn = as.double(wn), wp = double((p + 3) * p), aa = double(p * 
            		p), it.count = integer(2), info = integer(1), PACKAGE = "quantreg")
    if (z$info != 0) {  
	coefficients <- NA 
	} 
	else {   
		coefficients <- -z$wp[1:p] 
		}
    list(coefficients = coefficients)
}



# This function computes Extremal QR

rq.infer.extreme<- function(X,Y, tau=.1, subsample.fraction=.2, spacing=20, R=100, method = "fn", alpha = .10, ...) {

	if (tau>.5) {
		Y		<- -Y; 
		tau.e		<- 1-tau 
		}   
		else tau.e	<- tau

      
	n		<- length(Y); 	
	b		<- floor(subsample.fraction*n); 		
	m		<- (tau.e*n+spacing)/(tau.e*n);   	
	p 		<- dim(X)[2]
	tau.b.e		<- min(tau.e*n/b, tau.e+ (.5-tau.e)/3);
        if (tau.b.e == tau.e+ (.5-tau.e)/3 && b >= min(n/3, 1000)) 
		warning("tau may be non-extremal; results are not likely to differ from central inference");
 	muX		<- apply(X,2, mean) ;  
	betat		<- rq(Y~X[,-1], tau=tau.e, method= method)$coef ;   
	betatm		<- rq(Y~X[,-1], tau=m*tau.e, method= method)$coef ; 
	An		<- (m-1)*tau.e*sqrt(n/(tau.e*(1-tau.e)) )/(muX%*%(  betatm - betat   ) )
	betatb.e	<- rq(Y~X[,-1], tau=tau.b.e, method=method)$coef ;   
 	 

	Res		<- NULL;
	ss		<- matrix(sample(1:n, b * R, replace = T), b, R);
        m.b.e		<- (tau.b.e*b+spacing)/(tau.b.e*b);  

      
	 for(i in 1:R) { 		
		Xs		<- X[ss[,i],]; 
		Ys		<- Y[ss[,i]];
		if (method == "fn") {
			sub.betatb.e	<- rq.fit.fn.nostop(Xs,Ys, tau=tau.b.e)$coef ;  
			sub.betatbm.e	<- rq.fit.fn.nostop(Xs, Ys, tau=m.b.e*tau.b.e)$coef ;
			}	
			else {
				sub.betatb.e	<- coef(rq(Ys ~ Xs[ ,-1], tau=tau.b.e)) ;  
				sub.betatbm.e	<- coef(rq(Ys ~ Xs[ ,-1], tau=m.b.e*tau.b.e)) ;	
				}
 		Ab.e		<- (m.b.e-1)*tau.b.e*sqrt(b/(tau.b.e* (1-tau.b.e)) )/ ( muX%*%(sub.betatbm.e - sub.betatb.e) ) ;
		Sb.e		<-  Ab.e * (sub.betatb.e -  betatb.e);    
		Res		<- rbind(Res, Sb.e)
            	}



       object			<- NULL
       names(betat)		<- dimnames(X)[[2]]
       dimnames(Res)[[2]]	<- dimnames(X)[[2]]
       if(tau <=.5) {
       		object$betat	<- betat
       		object$BC.betat	<- betat - apply(Res, 2, quantile, .5, na.rm=TRUE)/An ;
       		object$ciL.e	<- betat - apply(Res, 2, quantile, 1 - alpha/2, na.rm=TRUE)/An ; 
       		object$ciU.e	<- betat - apply(Res, 2, quantile, alpha/2, na.rm=TRUE)/An ; 
	 	}
       		else {
	 		object$betat	<-   -betat
       			object$BC.betat	<- -(betat - apply(Res, 2, quantile, .5, na.rm=TRUE)/An) ;
       			object$ciL.e	<- -(betat - apply(Res, 2, quantile, alpha/2, na.rm=TRUE)/An) ; 
       			object$ciU.e	<- -(betat - apply(Res, 2, quantile, 1 - alpha/2, na.rm=TRUE)/An) ; 
       			}

       object$R		<- R-sum(is.na(Res[,1]));
       return(object);
	}



# Example

library(quantreg);


#	est<- rq(fmla.black, tau=.01, method="pfn")
#	summary.rq.extreme(est, subsample.fraction=4000/32000, R=100) 
#   	summary.rq(est, se="ker", mofn=2000)
#	summary.rq(est, se="boot", mofn=2000) 
#	summary.rq(est, se="boot", mofn=2000) 
#	summary.rq(est, se="ker”)



