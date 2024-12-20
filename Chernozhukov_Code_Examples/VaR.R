
# this section implements the estimation and inference results for the VaR example


library(quantreg);
library(foreign);
source("Y:/Extremes/Programs website/Updated/Programs/R-progs.R")

# Read VaR data set;

var_data 	<- read.table("Y:/Extremes/Programs website/Updated/Data/var_data.txt");
Y		<- var_data[1528:2527, 5];
Xt		<- var_data[1528:2527, 1:4];
X1.pos		<- (Xt[ ,2] >= 0) * Xt[ ,2];
X1.neg		<- -(Xt[ ,2] <= 0) * Xt[ ,2];
X2.pos		<- (Xt[ ,3] >= 0) * Xt[ ,3];
X2.neg		<- -(Xt[ ,3] <= 0) * Xt[ ,3];
X3.pos		<- (Xt[ ,4] >= 0) * Xt[ ,4];
X3.neg		<- -(Xt[ ,4] <= 0) * Xt[ ,4];
X1		<- Xt[ ,2];
X2		<- Xt[ ,3];

# section 1:  General Results


formula 	<- Y ~ X1.pos + X1.neg + X2.pos + X2.neg + X3.pos + X3.neg;  
p 		<- 7;
alpha		<- .10;


taus 		<- (1:199)/200; 
fit.central	<- array(0,c(p,3,length(taus)));
fit.extreme	<- array(0,c(p,3,length(taus)));


for(i in 1:length(taus)) {;
	fit			<- rq(formula, tau = taus[i]);
	central			<- summary.rq(fit, se = "ker");
	fit.central[,1,i]	<- central$coefficients[ ,1];
      	fit.central[,2,i] 	<- central$coefficients[ ,1] + qnorm(alpha/2) * central$coefficients[ ,2];
      	fit.central[,3,i] 	<- central$coefficients[ ,1] + qnorm(1 - alpha/2) * central$coefficients[ ,2];
	extreme			<- summary.rq.extreme(fit, subsample.fraction=.2, R=200, method = "br", alpha = alpha, spacing = 5+p);
	fit.extreme[,1,i]	<- extreme$coefficients[ ,5];
      	fit.extreme[,2,i] 	<- extreme$coefficients[ ,3];
      	fit.extreme[,3,i] 	<- extreme$coefficients[ ,4];
        };


postscript("Y:/Extremes/Programs website/Updated/Results/VaR_fig1_spacing5_asym.eps",horizontal=F,pointsize=11,width=6.0,height=8.0)


blab 	<- c("Intercept", "Spot Oil Return (+)", "Spot Oil Return (-)", "Market Return (+)", "Market Return (-)", "Own Return (+)", "Own Return (-)");
#blab 	<- c("Intercept", "Spot Oil Return", "Market Return", "Own Return (+)", "Own Return (-)");

minscale <- c(min(fit.extreme[1,2,]), min(fit.extreme[2:3,2,]), min(fit.extreme[2:3,2,]), min(fit.extreme[4:5,2,]), min(fit.extreme[4:5,2,]), min(fit.extreme[6:7,2,]), min(fit.extreme[6:7,2,]));
maxscale <- c(max(fit.extreme[1,3,]), max(fit.extreme[2:3,3,]), max(fit.extreme[2:3,3,]), max(fit.extreme[4:5,3,]), max(fit.extreme[4:5,3,]), max(fit.extreme[6:7,3,]), max(fit.extreme[6:7,3,]));

par(mfrow=c(4,2))

for(i in c(2:p,1)){
        b	<- fit.extreme[i,1,];
        b.p	<- fit.extreme[i,3,]; 
        b.m	<- fit.extreme[i,2,]; 
        plot( c(0,1 ), c(minscale[i], max(maxscale[i])), xlim =c(0, 1), type="n", xlab=expression(tau), ylab="Coefficient");
	title(paste(blab[i]),cex=.75);
       	lines(taus, smooth(b.p), col=4, lwd=1.5, lty=1);
       	lines(taus, smooth(b.m), col=4, lwd=1.5, lty=1);
       	abline(h=0);
       	# Plot Central Regions;
        b.c	<- fit.central[i,1,];
        b.p.c	<- fit.central[i,3,]; 
        b.m.c	<- fit.central[i,2,]; 
       	lines(taus, b.c,   col=1, lwd=1.5, lty = 1);
       	lines(taus, smooth(b.p.c), col=2, lty=2, lwd=1.5);
       	lines(taus, smooth(b.m.c), col=2, lty=2, lwd=1.5);       
 
        }

        plot( c(0,1), c(0,1), type="n", ylab="", xlab="", cex=.75, axes=FALSE, frame.plot=FALSE);
        title("", cex=.75);
        type.names<- c("QR Coefficient", "Extremal 90% CI ", "Central 90% CI ");
        legend(0, .9, legend = type.names, lty = c(1,1,2), col=c(1,4,2), bty="n", lwd=c(1.5,1.5,1.5), cex=1) ;

dev.off();





postscript("Y:/Extremes/Programs website/Updated/Results/VaR_fig2_spacing5_asym.eps",horizontal=F,pointsize=11,width=6.0,height=8.0)

rang	<- (taus <= .16);

blab 	<- c("Intercept", "Spot Oil Return (+)", "Spot Oil Return (-)", "Market Return (+)", "Market Return (-)", "Own Return (+)", "Own Return (-)")
blab 	<- c("Intercept", "Spot Oil Return", "Market Return", "Own Return (+)", "Own Return (-)");
blab 	<- c("Intercept", "Spot Oil Return (+)", "Spot Oil Return (-)", "Market Return (+)", "Market Return (-)", "Own Return (+)", "Own Return (-)");

minscale <- c(min(fit.extreme[1,2,rang]), min(fit.extreme[2:3,2,rang]), min(fit.extreme[2:3,2,rang]), min(fit.extreme[4:5,2,rang]), min(fit.extreme[4:5,2,rang]), min(fit.extreme[6:7,2,rang]), min(fit.extreme[6:7,2,rang]));
maxscale <- c(max(fit.extreme[1,3,rang]), max(fit.extreme[2:3,3,rang]), max(fit.extreme[2:3,3,rang]), max(fit.extreme[4:5,3,rang]), max(fit.extreme[4:5,3,rang]), max(fit.extreme[6:7,3,rang]), max(fit.extreme[6:7,3,rang]));

par(mfrow=c(4,2))


for(i in c(2:p,1)){
        b	<- fit.extreme[i,1,];
        b.p	<- fit.extreme[i,3,]; 
        b.m	<- fit.extreme[i,2,]; 
        plot( range(0, .16), c(minscale[i], max(maxscale[i])), xlim =c(0, .16), type="n", xlab=expression(tau), ylab="Coefficient");
	title(paste(blab[i]),cex=.75);
       	lines(taus, smooth(b),   col=1, lwd=1.5);
       	lines(taus, smooth(b.p), col=4, lwd=1.5, lty=1);
       	lines(taus, smooth(b.m), col=4, lwd=1.5, lty=1);
       	abline(h=0);
       	# Plot Central Regions;
        b.c	<- fit.central[i,1,];
        b.p.c	<- fit.central[i,3,]; 
        b.m.c	<- fit.central[i,2,]; 
       	lines(taus, smooth(b.p.c), col=2, lty=2, lwd=1.5);
       	lines(taus, smooth(b.m.c), col=2, lty=2, lwd=1.5);       
 
        }

        plot( c(0,1), c(0,1), type="n", ylab="", xlab="", cex=.75, axes=FALSE, frame.plot=FALSE);
        title("", cex=.75);
        type.names<- c("QR Coefficient - BC", "Extremal 90% CI ", "Central 90% CI ");
        legend(0, 1, legend = type.names, lty = c(1,1,2), col=c(1,4,2), bty="n", lwd=1.5, cex=1) ;

dev.off();


postscript("Y:/Extremes/Programs website/Updated/Results/VaR_fig3_spacing5_asym.eps",horizontal=F,pointsize=11,width=6.0,height=8.0)

rang	<- (taus >= .84);

blab 	<- c("Intercept", "Spot Oil Return (+)", "Spot Oil Return (-)", "Market Return (+)", "Market Return (-)", "Own Return (+)", "Own Return (-)")
blab 	<- c("Intercept", "Spot Oil Return", "Market Return", "Own Return (+)", "Own Return (-)");
blab 	<- c("Intercept", "Spot Oil Return (+)", "Spot Oil Return (-)", "Market Return (+)", "Market Return (-)", "Own Return (+)", "Own Return (-)");

minscale <- c(min(fit.extreme[1,2,rang]), min(fit.extreme[2:3,2,rang]), min(fit.extreme[2:3,2,rang]), min(fit.extreme[4:5,2,rang]), min(fit.extreme[4:5,2,rang]), min(fit.extreme[6:7,2,rang]), min(fit.extreme[6:7,2,rang]));
maxscale <- c(max(fit.extreme[1,3,rang]), max(fit.extreme[2:3,3,rang]), max(fit.extreme[2:3,3,rang]), max(fit.extreme[4:5,3,rang]), max(fit.extreme[4:5,3,rang]), max(fit.extreme[6:7,3,rang]), max(fit.extreme[6:7,3,rang]));

par(mfrow=c(4,2))


for(i in c(2:p,1)){
        b	<- fit.extreme[i,1,];
        b.p	<- fit.extreme[i,3,]; 
        b.m	<- fit.extreme[i,2,]; 
        plot( range(0.84, 1), c(minscale[i], max(maxscale[i])), xlim =c(0.84, 1), type="n", xlab=expression(tau), ylab="Coefficient");
	title(paste(blab[i]),cex=.75);
       	lines(taus, smooth(b),   col=1, lwd=1.5);
       	lines(taus, smooth(b.p), col=4, lwd=1.5, lty=1);
       	lines(taus, smooth(b.m), col=4, lwd=1.5, lty=1);
       	abline(h=0);
       	# Plot Central Regions;
        b.c	<- fit.central[i,1,];
        b.p.c	<- fit.central[i,3,]; 
        b.m.c	<- fit.central[i,2,]; 
       	lines(taus, smooth(b.p.c), col=2, lty=2, lwd=1.5);
       	lines(taus, smooth(b.m.c), col=2, lty=2, lwd=1.5);       
	 
        }

        plot( c(0,1), c(0,1), type="n", ylab="", xlab="", cex=.75, axes=FALSE, frame.plot=FALSE);
        title("", cex=.75);
        type.names<- c("QR Coefficient - BC", "Extremal 90% CI ", "Central 90% CI ");
        legend(0, 1, legend = type.names, lty = c(1,1,2), col=c(1, 4, 2), bty="n", lwd= 1.5, cex=1) ;

dev.off();




