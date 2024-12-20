
library(quantreg);
library(foreign);
source("Y:/Extremes/Programs website/Updated/Programs/R-progs.R")



# STAGE I. Preliminary Data Analysis

# Reponse -- infant birtwheight (in grams)

  nat 		<- read.dta("Y:/Extremes/Programs website/Updated/Data/natality.dta");
  attach(nat);

  weight	<- 1000*weight;
  m_wtgain2	<- (m_wtgain)^2;
  educ		<- ed_hs + 2*ed_smcol + 3*ed_col;
  mom_age.bar	<- rep(mean(mom_age), length(weight));
  m_wtgain.bar	<- rep(mean(m_wtgain), length(weight));
  mom_age.c	<- mom_age - mom_age.bar;
  mom_age.c2	<- mom_age2-(mom_age.bar)^2;
  m_wtgain.c	<- m_wtgain - m_wtgain.bar;
  m_wtgain.c2	<- m_wtgain2 - m_wtgain.bar^2;


postscript("Y:/Extremes/Programs website/Updated/Results/birthweight.densities.eps", pointsize=10, width=8, height=6, horizontal = FALSE, onefile = FALSE, family="AvantGarde");

	plot(density(weight[black==0],kernel="epanechnikov"), xlim=c(0, 5000), xlab="Infant Birthweight", ylab="Density", main=" ", lwd=2, col=1, type="n");
	lines(density(weight[black==0],kernel="epanechnikov", width=200), lty=3, lwd=2, col=2) ;
	lines(density(weight[black==1],kernel="epanechnikov", width=200), lty=2, lwd=2, col=1) ;
	legend(0 , .0008, legend=c("White", "Black"), lty=c(3,2), col=c(2,1));

dev.off();


# STAGE II. Explorations


# Here are Basis Regressions that Show huge discrepancy between White and Black Mothers:


# Choice of Specification: following Koenker and Hallock -- but more parsimonious.


fmla.black	<-  weight[black==1] ~ married[black==1] + boy[black==1] +  mom_age.c[black==1] + mom_age.c2[black==1]  + tri2[black==1]  + tri3[black==1]  + novisit[black==1]  + educ[black==1]  + smoke[black==1] +  cigsper[black==1]  +  m_wtgain.c[black==1]  + m_wtgain.c2[black==1];
fmla.white	<-  weight[black==0] ~ married[black==0] + boy[black==0] +  mom_age.c[black==0] + mom_age.c2[black==0]  + tri2[black==0]  + tri3[black==0]  + novisit[black==0]  + educ[black==0]  + smoke[black==0] +  cigsper[black==0]  +  m_wtgain.c[black==0]  + m_wtgain.c2[black==0];

fmla.black	<-  weight[black==1] ~ married[black==1] + boy[black==1] +  mom_age.c[black==1] + mom_age.c2[black==1]  + tri2[black==1]  + tri3[black==1]  + novisit[black==1]  + educ[black==1]  + smoke[black==1] +  cigsper[black==1] ;
fmla.white	<-  weight[black==0] ~ married[black==0] + boy[black==0] +  mom_age.c[black==0] + mom_age.c2[black==0]  + tri2[black==0]  + tri3[black==0]  + novisit[black==0]  + educ[black==0]  + smoke[black==0] +  cigsper[black==0] ;


# STAGE III.  Compute Estimates and Confidence Intervals Using Non-Exremal Inference



taus		<- c( seq(.0003, .01, by=.0006), .01, .015, .025, .03,  .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, .985, .99, seq(.995, .9997, by=.0006) );
fit.black.ols 	<- summary(lm(fmla.black))$coefficients;
p 		<- nrow(fit.black.ols);
fit.black 	<- array("NaN",c(p,4,length(taus)));
taus.na		<- NULL;
for(i in 1:length(taus)) {;
        f 		<- rq(fmla.black, taus[i], method="fn");
        if(sum(is.na(f$coef))>=1) {;  
		print(c(i, "not available"));
		taus.na		<- c(taus.na, i);
		};
	if(sum(is.na(f$coef))==0) { 
		fit.black[,,i]	<- summary.rq(f, se="ker")$coefficients; 
		};
        };


# numbers 6 (tau = .0033) and 31 (tau = .8) have not computed; so drop them;

if (!is.null(taus.na)) {;
	taus		<- taus[-taus.na];
	fit.black	<- fit.black[,,-taus.na];
	};



# STAGE IV.  Computed Estimates and Cofidence Intervals for Extremal Quantiles Using Both Extremal Methods


taus.e			<- taus;
l			<- length(taus);
fit.black.extreme 	<- array(0,c(p,5,length(taus)));
taus.na			<- NULL;

for(i in 1:length(taus) ) {;
	print(i);
        f 			<- rq(fmla.black, taus.e[i], method="fn");
	f.extreme		<- summary.rq.extreme(f, subsample.fraction=4000/32000, R=500, alpha = .10, spacing = p+5);
        if(sum(is.na(f.extreme$coef[,2]))>=1) {;  
		print(c(i, "not available"));
		taus.na		<- c(taus.na, i);
		};
        fit.black.extreme[,,i] 	<- f.extreme$coefficients;
        };

# now numbers 21 (tau = .05) and 31 (tau = .95) are missing for some reason; so drop them;


if (!is.null(taus.na)) {;
	taus			<- taus[-taus.na];
	fit.black		<- fit.black[,,-taus.na];
	fit.black.extreme	<- fit.black.extreme[,,-taus.na];
	};

taus.final		<- taus
alpha			<- .10;

# LOWER TAIL


 postscript("Y:/Extremes/Programs website/Updated/Results/natality_lower_tail2.spacing5.smooth.eps", width = 8, height = 10,
                horizontal = FALSE, onefile = FALSE, family="AvantGarde");

	p 	<- dim(fit.black.extreme)[1];
	blab	<- c("Centercept", "Married", "Boy", "Mother's Age", "Mother's Age^2", "Prenatal Second", 
	 		"Prenatal Third", "No Prenatal",  "Education", 
 	 		"Smoker", "Cigarette's/Day");
	
        taus	<- taus.final;
	fit	<- fit.black.extreme;
	fit2	<- fit.black; 

	par(mfrow=c(4,3), lab=c(5,5,5));

	rang 	<- (taus <= .025);

	for(i in c(1,2,3,4,5,6,7,8,9,10,11)){;
		b		<- fit[i,5,];
        	b.p		<- fit[i,3,];
        	b.m		<- fit[i,4,];
       	 	plot(c(0,.025),range(c(b.m[rang],b.p[rang])),type="n",xlab = expression(tau),ylab="coef",cex=.75);
		title(paste(blab[i]), cex=.75);
        	lines(taus, smooth(b),   col=1, lwd=1.5);
        	lines(taus, smooth(b.p), col=4, lwd=1.5, lty=1);
        	lines(taus, smooth(b.m), col=4, lwd=1.5, lty=1);
        	abline(h=0);
        	# Plot Central Regions
		b.c		<- fit[i,1,];
		b.p		<- b.c + qnorm(1 - alpha/2)*as.numeric(fit2[i,2,]);
        	b.m		<- b.c + qnorm(alpha/2)*as.numeric(fit2[i,2,]);
        	lines(taus, smooth(b.p), col=2, lty=2, lwd=2);
        	lines(taus, smooth(b.m), col=2, lty=2, lwd=2) ;      
        };
        plot( c(0,1), c(0,1), type="n", ylab="", xlab="", cex=.75, axes=FALSE, frame.plot=FALSE);
        title("", cex=.75);
        type.names	<- c("QR Coefficient - BC", "Extremal 90% CI ", "Central 90% CI ");
        legend(0, .9, legend = type.names, lty = c(1,1,2), col=c(1,4,2), bty="n", lwd=1.5, cex=1);

dev.off();





# MIDDLE 


postscript("Y:/Extremes/Programs website/Updated/Results/natality_middle2.spacing5.smooth.eps", width = 8, height = 10,
                horizontal = FALSE, onefile = FALSE, family="AvantGarde");
	p 	<- dim(fit.black.extreme)[1];
	blab	<- c("Centercept", "Married", "Boy", "Mother's Age", "Mother's Age^2", "Prenatal Second", 
	 		"Prenatal Third", "No Prenatal",  "Education", 
 	 		"Smoker", "Cigarette's/Day");
	
        par(mfrow=c(4,3), lab=c(10,10,10));

	rang <- (taus >= .02 & taus <= .98);

	for(i in c(1,2,3,4,5,6,7,8,9,10,11)){;
		b		<- as.numeric(fit[i,1,]);
        	b.p		<- as.numeric(fit[i,3,]);
        	b.m		<- as.numeric(fit[i,4,]);
       	 	plot(c(0.07,.93),range(c(b.m[rang],b.p[rang])),type="n",xlab=expression(tau),ylab="coef",cex=.75);
		title(paste(blab[i]),cex=.75);
        	lines(taus, smooth(b.p), col=4, lwd=1.5, lty=1);
        	lines(taus, smooth(b.m), col=4, lwd=1.5, lty=1);
        	abline(h=0);
        	# Plot Central Regions;
		b.c		<- as.numeric(fit2[i,1,]);
		b.p		<- b.c + 1.69*as.numeric(fit2[i,2,]);
        	b.m		<- b.c - 1.69*as.numeric(fit2[i,2,]);
        	lines(taus, smooth(b.c), col=1, lwd=2);
        	lines(taus, smooth(b.p), col=2, lty=2, lwd=2);
        	lines(taus, smooth(b.m), col=2, lty=2, lwd=2);       
        };
        plot( c(0,1), c(0,1), type="n", ylab="", xlab="", cex=.75, axes=FALSE, frame.plot=FALSE);
        title("", cex=.75);
        type.names<- c("QR Coefficient", "Extremal 90% CI ", "Central 90% CI ");
        legend(0, .9, legend = type.names, lty = c(1,1,2), col=c(1,4,2), bty="n", lwd=1.5, cex=1) ;

dev.off();

