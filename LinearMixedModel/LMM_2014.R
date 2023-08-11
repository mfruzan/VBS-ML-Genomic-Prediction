require(asreml)
pv <- read.csv('Grain.yield_adjusted.csv', stringsAsFactors=F)
marker <- read.csv("genetic_marker_matrix_M.csv")
rownames(marker) <- pv$Genotype
trainID <-read.table("split/2014/01/trainID.txt", header=FALSE, stringsAsFactors=FALSE)
ID.VEC <- as.numeric(trainID[,1])
ID.VEC <- ID.VEC+1
pheno <- pv[ID.VEC,]
#mark <- marker[rownames(marker) %in% pheno$Genotype,]
mark <- marker[ID.VEC,]
pheno$Genotype <- as.factor(pheno$Genotype)
Sys.time()
kmat <- crossprod(t(mark))
kbar <- mean(diag(kmat))
kmat <- kmat/kbar
Sys.time()

kmat.inv <- solve(kmat)
attr(kmat.inv, "INVERSE") <- TRUE
Sys.time()
## transformation matrix
Sys.time()
asr <- asreml(Adj.Grain.Yield ~ 1, random = ~ vm(Genotype, kmat.inv), data=pheno, na.action=na.method(x='include'), maxit=30,  workspace="28gb")
#save(asr, file="asr4.RData")
#load("asr4.RData")

Sys.time()
 
#testing 
trans <- (t(mark) %*% kmat.inv)/kbar
testID <-read.table("solit/2014/01/testID.txt", header=FALSE, stringsAsFactors=FALSE)
TEST.ID.VEC <- as.numeric(testID[,1])
TEST.ID.VEC <- TEST.ID.VEC+1

gblups <- asr$coefficients$random
gnams <- sapply(strsplit(rownames(gblups), "_"), "[", 2)
pred <- cbind.data.frame(Genotype = gnams, gblups = gblups[,1], gpred = gblups[,1] + asr$coefficients$fixed[,1])
rownames(pred) <- NULL
## obtain marker effects from training set
mblups <- trans %*% pred$gblups
#test.mark <- marker[rownames(marker) %in% pv[TEST.ID.VEC,"Genotype"],]
test.mark <- marker[TEST.ID.VEC,]

test.gblups <- as.matrix(test.mark) %*% mblups
test.gpred <- test.gblups + asr$coefficients$fixed[,1]
test.observed <- pv[pv$Genotype %in% rownames(test.gpred), c("Genotype","Adj.Grain.Yield")]
#reorder test.gpred
test.gpred <- test.gpred[test.observed$Genotype ,]
#relative absolute error
sum(abs(test.gpred-test.observed[,2])/test.observed[,2])/length(test.observed[,2])
write.table(test.gpred, file="2014_1.txt", sep='\t', quote = FALSE,  row.names = T, append = F)
#Split 1: 0.07387
#Split 2: 0.07211
#Split 3: 0.0744
#Split 4: 0.0731

#Mean of absolute error
sum(abs(test.gpred-test.observed[,2]))/length(test.gpred)
#Split 1 369.00
#Split 2: 358.85
#split 3: 369.68

 


