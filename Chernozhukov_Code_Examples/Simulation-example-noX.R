

#Simulation Example for The Paper:
#Reproduction
## Part I: EXtreme Rank Approximation

set.seed(88)

SP	<- 200;
N	R<- 10000;

#	Set-up 1: Fairly Low Quantile: tau=.025
#	tau<- .025 ;
#	k<- .025 * SP = 5;
# 	To see that it works type at S prompt quantile(1:100, .02). It should
# 	return "4.98", # i.e. the fifth order statistics.
 
qe5	<- rep(0,NR);
for (i in 1:NR) {	
	Y	<- rcauchy(SP); 		
	qe5[i]	<- quantile(Y, .02)
	}

qex5	<-  rep(0,NR); 
for (i in 1:NR) { 
	ee	<- rexp (5);    
	qex5[i]	<- -(qcauchy(1/SP))*(sum (ee))^{-1} 
	}

qet5	<- qe5 - qcauchy(.025)  # center at the population quantiles
qext5	<- qex5 -  (-qcauchy(1/SP))*(5)^{-1} #center around c(k)





#	Set-up 2: Low Quantile: tau=.2
#	tau<- .2 ;
#	k<- .2 * SP = 40;

qe40	<- rep(0,NR);
for (i in 1:NR) {	
	Y<- rcauchy(SP); 		
	qe40[i]<- quantile(Y, .195)
	}
	
qex40	<-  rep(0,NR); 
for (i in 1:NR) { 
	ee	<- rexp (40);    
	qex40[i]<- -(qcauchy(1/SP))*(sum (ee))^{-1} 	
	}

# Centering 
qet40	<- qe40 - qcauchy(.2)  # center at the population quantiles
qext40	<- qex40 -  (-qcauchy(1/SP))*(40)^{-1} #center around c(k)



#	Set-up 2: Low Quantile: tau=.3
#	tau<- .3 ;
#	k<- .3 * SP = 60;

qe60	<- rep(0,NR);
for (i in 1:NR) {	
	Y<- rcauchy(SP); 		
	qe60[i]<- quantile(Y, .297)
	}
	
qex60	<-  rep(0,NR); 
for (i in 1:NR) { 
	ee<- rexp (60);    
	qex60[i]<- -(qcauchy(1/SP))*(sum (ee))^{-1} 
	}

# Centering 
qet60	<- qe60 - qcauchy(.3)  # center at the population quantiles
qext60	<- qex60 -  (-qcauchy(1/SP))*(60)^{-1} #center around c(k)


# produce QQ-plots

postscript("Y:/Extremes/Programs website/Updated/Results/fig.noX.eps",horizontal=F,pointsize=12,width=8.0,height=4.0)

par(mfrow=c(1,3))

plot( quantile(qet5, ppoints(qet5, .02)), quantile (qet5, ppoints(qet5, .02)) , type="l" , xlim=c(-110,20), ylim =c(-110, 20),
xlab="", ylab="" , lwd = 1.5  )
title(main=expression(bold(paste("A. ",tau, " = 0.025, T = 200, ",tau,"T = 5"))) )

lines( quantile(qet5, ppoints(qet5, .02)), quantile (-qext5, ppoints(qet5, .02)) , type="l", lty=3, col=4 , lwd = 3)

lines(  quantile(qet5, ppoints(qet5, .02)),  qnorm(  ppoints(qet5, .02), 0,  sqrt(.025*.975/SP)/ dcauchy(qcauchy(.025) )), lty=2,  col=2, lwd = 1.5) 


plot( quantile(qet40, ppoints(qet40, .05)), quantile (qet40, ppoints(qet40, .05)) , type="l" , xlim= c(-1.5, 1.5) , ylim =c(-1.5, 1.5),
xlab="", ylab="" , lwd = 1.5 )
title(main=expression(bold(paste("B. ",tau, " = 0.2, T = 200, ",tau,"T = 40"))) )

lines( quantile(qet40, ppoints(qet40, .05)), quantile (-qext40, ppoints(qet40, .05)) , type="l", lty=3, col=4, lwd = 3 )

lines(  quantile(qet40, ppoints(qet40, .05)),  qnorm(  ppoints(qet40, .05), 0,  sqrt(.2*.8/SP)/ dcauchy(qcauchy(.2) )), lty=2,  col=2, lwd = 1.5) 


plot( quantile(qet60, ppoints(qet60, .01)), quantile (qet60, ppoints(qet60, .01)) , type="l" ,  xlim=c(-.75,.75), ylim =c(-.75, .75),
xlab="", ylab="", lwd = 1.5 )
title(main=expression(bold(paste("C. ",tau, " = 0.3, T = 200, ",tau,"T = 60"))) )

lines( quantile(qet60, ppoints(qet60, .01)), quantile (-qext60, ppoints(qet60, .01)) , type="l", lty=3, col=4, lwd = 3 )

lines(  quantile(qet60, ppoints(qet60, .01)),  qnorm(  ppoints(qet60, .01), 0,  sqrt(.3*.7/SP)/ dcauchy(qcauchy(.3) )), lty=2,  col=2, lwd = 1.5) 



dev.off()

################

