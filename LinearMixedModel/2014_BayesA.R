rm(list=ls())
setwd("/mnt/NFS/mario/projects/AGT-GS/challenge2")

require(BGLR)

pv <- read.csv('Grain.yield_adjusted.csv', stringsAsFactors=F)
marker <- read.csv("genetic_marker_matrix_M.csv")
rownames(marker) <- pv$Genotype
y <- pv$Adj.Grain.Yield

testID <- read.table("2014_splits/04/testID.txt", header=FALSE, stringsAsFactors=FALSE)
ID.VEC <- as.numeric(testID[,1])
ID.VEC <- ID.VEC+1
y[ID.VEC] <- NA

marker[,1:ncol(marker)] <- lapply(marker[,1:ncol(marker)],as.numeric)

thin <- 3
saveAt <- ''
S0 <- NULL
weights <- NULL
R2 <- 0.5
nIter  <- 500
burnIn <- 100

ETA<-list(list(X=marker,model='BayesA'))
fit <- BGLR(y=y,ETA=ETA,nIter=nIter,burnIn=burnIn,thin=thin,saveAt=saveAt,df0=5,S0=S0,weights=weights,R2=R2)
  
y <- pv$Adj.Grain.Yield
#relative absolute error
sum(abs(fit$yHat[ID.VEC]- y[ID.VEC])/y[ID.VEC])/length(ID.VEC)
cor(fit$yHat[ID.VEC], y[ID.VEC])
cor(fit$yHat[ID.VEC], y[ID.VEC], method="spearman")

#split1 : 0.07286211 (0.5365304)
#split2 : 0.07161784 (0.4994519)
#split 3 : 0.0737769 (0.5095261)
#split 4 : 0.07254431 ( 0.48755)

dfout <- data.frame(genotype=pv[ID.VEC,"Genotype"], pred=fit$yHat[ID.VEC], obs=y[ID.VEC])
dfout <- dfout[order(dfout$genotype),]
write.csv(dfout, file="2014_4_BA.csv",  quote = FALSE,  row.names = F)
