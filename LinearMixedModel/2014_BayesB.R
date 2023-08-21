rm(list=ls())
setwd("/mnt/NFS/mario/projects/AGT-GS/challenge2")

require(BGLR)
pv <- read.csv('Grain.yield_adjusted.csv', stringsAsFactors=F)
marker <- read.csv("genetic_marker_matrix_M.csv")
rownames(marker) <- pv$Genotype
y <- pv$Adj.Grain.Yield

testID <-read.table("2014_splits/02/testID.txt", header=FALSE, stringsAsFactors=FALSE)
ID.VEC <- as.numeric(testID[,1])
ID.VEC <- ID.VEC+1
y[ID.VEC] <- NA

marker <- marker[,-1]
marker[,1:ncol(marker)] <- lapply(marker[,1:ncol(marker)],as.numeric)

thin <- 10
saveAt <- ''
S0 <- NULL
weights <- NULL
R2 <- 0.5
nIter  <- 5000
burnIn <- 2500

ETA<-list(list(X=marker,model='BayesB',probIn=0.05))
fit <- BGLR(y=y,ETA=ETA,nIter=nIter,burnIn=burnIn,thin=thin,saveAt=saveAt,df0=5,S0=S0,weights=weights,R2=R2)  
y <- pv$Adj.Grain.Yield

#relative absolute error

sum(abs(fit$yHat[ID.VEC]- y[ID.VEC])/y[ID.VEC])/length(ID.VEC)
cor(fit$yHat[ID.VEC], y[ID.VEC])
cor(fit$yHat[ID.VEC], y[ID.VEC], method="spearman")

#split1 : 0.07204854 (0.545329)
#split2 : 0.07089315  (0.5135163)
#split3 : 0.07364602 (0.5159376) 
#split4 : 0.07210284 (0.4951197)

dfout <- data.frame(genotype=pv[ID.VEC,"Genotype"], pred=fit$yHat[ID.VEC], obs=y[ID.VEC])
dfout <- dfout[order(dfout$genotype),]
write.csv(dfout, file="2014_2_BB.csv",  quote = FALSE,  row.names = F)

