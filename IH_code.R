#Code for Predicting the Outcome of Ischemic Hepatitis with Real Patient Data Using Machine Learning Tools

#LIBRARIES----------------------------------------------------------------------
#libraries needed for functions used
library(readxl) #read_excel
library(caret) #createDataPartion, confusionMatrix
library(tree) #tree
library(randomForest) #randomForest
library(BART) #pbart
library(neuralnet) #neuralnet
library(smotefamily) #SMOTE
library(pROC) #roc, coords, ci.coords, ci.auc
library(ROCR) #prediction, performance
library(ggplot2) #ggplot


#DATA---------------------------------------------------------------------------
data <- read_excel("IH_cleaned_data.xlsx") #read in data file


#AST/ALT------------------------------------------------------------------------
r <- cor(data$ALT_Peak, data$AST_Peak) #correlation coefficient between ALT & AST
AST.ALT <- data$AST_Peak/data$ALT_Peak #make AST/ALT ratio variable

vars <- data.frame(AST.ALT, data$Blrb_Peak, data$Crtn_Peak, data$INR_Peak, data$Outcome)
colnames(vars)[2] <- "Crtn.Peak"
colnames(vars)[3] <- "INR.Peak"
colnames(vars)[4] <- "Blrb.Peak"
colnames(vars)[5] <- "Outcome"


#SUBSETS------------------------------------------------------------------------
set.seed(1)
index <- createDataPartition(unlist(vars[,5]), p=0.8, list=FALSE, times=1)
train.df <- vars[index,] #training set (80%)
test <- vars[-index,] #testing set (20% validation)

y.test <- test$Outcome #test set response


#SCALE--------------------------------------------------------------------------
train.scaled <- as.data.frame(lapply(train.df[,-5], scale)) #normalize training predictors
train <- cbind(train.scaled, train.df[,5]) #combine normalized training vars with response var

colnames(train)[5] <- "Outcome"


#LOGISTIC REGRESSION------------------------------------------------------------
full.model <- glm(Outcome~., data=train, family="binomial") #full logistic model
summary(full.model) 

null.model <- glm(Outcome~1, data=train, family="binomial") #empty logistic model
logreg <- step(null.model, scope=list(upper=full.model), direction="both", test="Chisq", trace=F) #model selection
summary(logreg) #stepwise model

prob.logreg <- predict(logreg, newdata=test, type="response") #probability of test set outcome
yhat.logreg <- ifelse(prob.logreg >= 0.5, 1, 0) #prediction of test set outcome

caret::confusionMatrix(data=factor(yhat.logreg, levels=c(0,1)), reference=factor(y.test, levels=c(0,1)), positive='1') #eval metrics
roc.logreg <- pROC::roc(response=y.test, predictor=prob.logreg, levels=c(0,1), direction="<")
  plot(roc.logreg) #ROC sensitivity vs specificity
  print(roc.logreg) #AUC
  ci.auc(roc.logreg) #AUC CI

print(pROC::coords(roc.logreg, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics at threshold 0.5
print(pROC::ci.coords(roc.logreg, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics CI

#ROC fpr vs tpr
logreg.pred <- ROCR::prediction(prob.logreg, y.test)
logreg.roc <- ROCR::performance(logreg.pred, "tpr", "fpr")
  logreg.auc <- ROCR::performance(logreg.pred, "auc")

plot(logreg.roc) #ROC
  print(logreg.auc@y.values[[1]]) #AUC


#REGRESSION TREE----------------------------------------------------------------
full.tree <- tree(Outcome~., data=train) #full regression tree
plot(full.tree) #plot full regression tree
text(full.tree, pretty=0) #add labels to plot

#set.seed(1)
cv <- cv.tree(full.tree, K=5) #5-fold cross-validation
plot(cv$size, cv$dev, type='b') #determine "best" number of nodes for pruning (want: min)

tree <- prune.tree(full.tree, best=3) #pruned regression tree
plot(tree) #plot pruned regression tree shape
text(tree, pretty=0) #add labels to plot

prob.tree <- predict(tree, newdata=test) #probability of test set outcome
yhat.tree <- ifelse(prob.tree >= 0.5, 1, 0) #prediction of test set outcome

caret::confusionMatrix(data=factor(yhat.tree, levels=c(0,1)), reference=factor(y.test, levels=c(0,1)), positive='1') #eval metrics
roc.tree <- pROC::roc(response=y.test, predictor=prob.tree, levels=c(0,1), direction="<")
  plot(roc.tree) #ROC sensitivity vs specificity
  print(roc.tree) #AUC
  ci.auc(roc.tree) #AUC CI

print(pROC::coords(roc.tree, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity", "auc"), transpose=F)) #point metrics at threshold 0.5
print(pROC::ci.coords(roc.tree, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics CI

tree.pred <- ROCR::prediction(prob.tree, y.test)
tree.roc <- ROCR::performance(tree.pred, "tpr", "fpr")
  tree.auc <- performance(tree.pred, "auc")

plot(tree.roc) #ROC fpr vs tpr
  print(tree.auc@y.values[[1]]) #AUC

#RANDOM FOREST------------------------------------------------------------------
#set.seed(1)
rf <- randomForest(factor(Outcome)~., data=train, ntree=1000, mtry=3) #baseline RF

pred.rf <- predict(rf, newdata=test, type="prob")
prob.rf <- pred.rf[,2] #probability of 1 (survival)
yhat.rf <- ifelse(prob.rf >= 0.5, 1, 0) #prediction of test set outcome

rf$importance #variable importance

caret::confusionMatrix(data=factor(yhat.rf, levels=c(0,1)), reference=factor(y.test, levels=c(0,1)), positive='1') #eval metrics
roc.rf <- pROC::roc(response=y.test, predictor=prob.rf, levels=c(0,1), direction="<")
  plot(roc.rf) #ROC sensitivity vs specificity
  print(roc.rf) #AUC
  ci.auc(roc.rf) #AUC CI

print(pROC::coords(roc.rf, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity", "auc"), transpose=F)) #point metrics at threshold 0.5
print(pROC::ci.coords(roc.rf, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics CI

rf.pred <- ROCR::prediction(prob.rf, y.test)
rf.roc <- ROCR::performance(rf.pred, "tpr", "fpr")
  rf.auc <- performance(rf.pred, "auc")

plot(rf.roc) #ROC #plot fpr vs tpr
  print(rf.auc@y.values[[1]]) #AUC


#BART---------------------------------------------------------------------------
x.train <- train[, 1:4] #train set predictors
y.train <- train$Outcome #train set response
x.test <- test[, 1:4] #test set predict

#set.seed(1)
bart.fit <- pbart(x.train, y.train, x.test) #model

order <- order(bart.fit$varcount.mean, decreasing=T) #avg variable count in decreasing order
bart.fit$varcount.mean[order] #variable importance

prob.bart <- bart.fit$prob.test.mean #probability of test set outcome
yhat.bart <- ifelse(prob.bart >= 0.5, 1, 0) #prediction of test set outcome

caret::confusionMatrix(data=factor(yhat.bart, levels=c(0,1)), reference=factor(y.test, levels=c(0,1)), positive='1') #eval metrics
roc.bart <- pROC::roc(response=y.test, predictor=prob.bart, levels=c(0,1), direction="<")
  plot(roc.bart) #ROC sensitivity vs specificity
  print(roc.bart) #AUC
  ci.auc(roc.bart) #AUC CI

print(pROC::coords(roc.bart, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity", "auc"), transpose=F)) #point metrics at threshold 0.5
print(pROC::ci.coords(roc.bart, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics CI

bart.pred <- ROCR::prediction(prob.bart, y.test)
bart.roc <- ROCR::performance(bart.pred, "tpr", "fpr")
  bart.auc <- performance(bart.pred, "auc")

plot(bart.roc) #ROC fpr vs tpr
  print(bart.auc@y.values[[1]]) #AUC


#NEURAL NETWORK-----------------------------------------------------------------
#set.seed(1)
nn <- neuralnet(Outcome~AST.ALT+Blrb.Peak+Crtn.Peak+INR.Peak,
                data=train, hidden=2, rep=3, linear.output=FALSE, threshold=0.001)
plot(nn, rep="best") #model

prob.nn <- predict(nn, newdata=test) #probability of test set outcome
yhat.nn <- ifelse(prob.nn >= 0.5, 1, 0) #prediction of test set outcome

caret::confusionMatrix(data=factor(yhat.nn, levels=c(0,1)), reference=factor(y.test, levels=c(0,1)), positive='1') #eval metrics
roc.nn <- pROC::roc(response=y.test, predictor=prob.nn, levels=c(0,1), direction="<")
  plot(roc.nn) #ROC sensitivity vs specificity
  print(roc.nn) #AUC
  ci.auc(roc.nn) #AUC CI

print(pROC::coords(roc.nn, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity", "auc"), transpose=F)) #point metrics at threshold 0.5
print(pROC::ci.coords(roc.nn, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #boostrap CIs at threshold 0.5

nn.pred <- ROCR::prediction(prob.nn, y.test)
nn.roc <- ROCR::performance(nn.pred, "tpr", "fpr")
  nn.auc <- performance(nn.pred, "auc")

plot(nn.roc) #ROC fpr vs tpr
  print(nn.auc@y.values[[1]]) #AUC


#ROC CURVES---------------------------------------------------------------------
plot(logreg.roc, col="#F9AE9F", lty=1, lwd=5, family="serif", cex.lab=1.3, cex.axis=1.5)
plot(tree.roc, col="#FADF57", lty=1, lwd=5, add=TRUE)
plot(rf.roc, col="#AACD9D", lty=1, lwd=5, add=TRUE)
plot(bart.roc, col="#98C1F1", lty=1, lwd=5, add=TRUE)
plot(nn.roc, col="gray", lty=1, lwd=5, add=TRUE)
abline(0,1)
legend("topleft", inset=0.025, cex=1.3, text.font=6,
       legend=c("Logistic Regression", "Regression Tree", "Random Forest", "BART", "Neural Network"),
       col=c("#F9AE9F","#FADF57","#AACD9D","#98C1F1","gray"), lty=(1), lwd=(5))


#SMOTE--------------------------------------------------------------------------
prop.table(table(train$Outcome)) #original response proportion of training set

n1 <- table(train$Outcome)['1'] #number of "1" responses
n0 <- table(train$Outcome)['0'] #number of "0" responses
r0 <- 0.50 #ideal proportion of responses
ntimes <- ((1-r0)/r0)*(n1/n0)-1 #formula

train[6] <- 1:length(train$Outcome) #add index to original train data
colnames(train)[6] <-  "Index" #name index for original train data

#set.seed(1)
SMOTE <- SMOTE(X=train, target=train$Outcome, K=5, dup_size=ntimes) #apply algorithm
SMOTE.DATA <- SMOTE$data #new dataset
SMOTE.data <- SMOTE.DATA[,-7] #remove duplicated class
  #order by original data index to easily identify synthetic observations  
  SMOTE.data <- SMOTE.data[order(unlist(SMOTE.data[,6])),] 
smote.train <- SMOTE.data[,-6] #remove index for completeness

prop.table(table(smote.train$Outcome)) #SMOTE response proportion

#plot data responses
ggplot(train, aes(x=Blrb.Peak, y=Crtn.Peak, color=factor(Outcome))) +
  geom_point() + scale_color_manual(values=c('#F9AE9F','#98C1F1')) +
  ggtitle("Data") +  theme(plot.title=element_text(hjust=0.5))
#plot SMOTE data responses (can see increase in minority "0")
ggplot(smote.train, aes(x=Blrb.Peak, y=Crtn.Peak, color=factor(Outcome))) +
  geom_point() +scale_color_manual(values=c('#F9AE9F','#98C1F1')) +
  ggtitle("SMOTE Data") +  theme(plot.title=element_text(hjust=0.5))


#SMOTE LOGISTIC REGRESSION------------------------------------------------------
smote.full.model <- glm(Outcome~., data=smote.train, family="binomial") #full logistic model
summary(smote.full.model)

smote.null.model <- glm(Outcome~1, data=smote.train, family="binomial") #empty logistic model
smote.logreg <- step(smote.null.model, scope=list(upper=smote.full.model), direction="both", test="Chisq", trace=F) #model selection
summary(smote.logreg) #stepwise model

smote.prob.logreg <- predict(smote.logreg, newdata=test, type="response") #probability of test set response
smote.yhat.logreg <- ifelse(smote.prob.logreg >= 0.5, 1, 0) #prediction of test set outcome

caret::confusionMatrix(data=factor(smote.yhat.logreg, levels=c(0,1)), reference=factor(y.test, levels=c(0,1)), positive='1') #eval metrics
smote.roc.logreg <- pROC::roc(response=y.test, predictor=smote.prob.logreg, levels=c(0,1), direction="<")
  plot(smote.roc.logreg) #ROC specificity vs sensitivity
  print(smote.roc.logreg) #AUC
  ci.auc(smote.roc.logreg) #AUC CI

print(pROC::coords(smote.roc.logreg, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics at threshold 0.5
print(pROC::ci.coords(smote.roc.logreg, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics CI

logreg.pred.smote <- ROCR::prediction(smote.prob.logreg, y.test)
logreg.roc.smote <- ROCR::performance(logreg.pred.smote, "tpr", "fpr")
  logreg.auc.smote <- ROCR::performance(logreg.pred.smote, "auc")

plot(logreg.roc.smote) #ROC fpr vs tpr
  print(logreg.auc.smote@y.values[[1]]) #AUC


#SMOTE REGRESSION TREE----------------------------------------------------------
smote.full.tree <- tree(Outcome~., data=smote.train) #full regression tree
plot(smote.full.tree) #plot full regression tree
text(smote.full.tree, pretty=0) #add labels to plot

#set.seed(1)
cv <- cv.tree(smote.full.tree, K=5) #5-fold cross-validation
plot(cv$size, cv$dev, type='b') #determine "best" number of nodes for pruning (want: min)

smote.tree <- prune.tree(smote.full.tree, best=6) #pruned regression tree
plot(smote.tree) #plot pruned regression tree
text(smote.tree, pretty=0) #add labels to plot

smote.prob.tree <- predict(smote.tree, newdata=test) #probability of test set outcome
smote.yhat.tree <- ifelse(smote.prob.tree >= 0.5, 1, 0) #prediction of test set outcome

caret::confusionMatrix(data=factor(smote.yhat.tree, levels=c(0,1)), reference=factor(y.test, levels=c(0,1)), positive='1') #eval metrics
smote.roc.tree <- pROC::roc(response=y.test, predictor=smote.prob.tree, levels=c(0,1), direction="<")
  plot(smote.roc.tree) #ROC specificity vs sensitivity
  print(smote.roc.tree) #AUC
  ci.auc(smote.roc.tree) #AUC CI

print(pROC::coords(smote.roc.tree, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity", "auc"), transpose=F)) #point metrics at threshold 0.5
print(pROC::ci.coords(smote.roc.tree, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics CI

tree.pred.smote <- ROCR::prediction(smote.prob.tree, y.test)
tree.roc.smote <- ROCR::performance(tree.pred.smote, "tpr", "fpr")
  tree.auc.smote <- performance(tree.pred.smote, "auc")

plot(tree.roc.smote) #ROC fpr vs tpr
  print(tree.auc.smote@y.values[[1]]) #AUC


#SMOTE RANDOM FOREST------------------------------------------------------------
smote.rf <- randomForest(factor(Outcome)~., data=smote.train, ntree=1000, mtry=3) #baseline RF

smote.rf$importance #variable importance

smote.pred.rf <- predict(smote.rf, newdata=test, type="prob")
smote.prob.rf <- smote.pred.rf[,2] #probability of 1 (survival)
smote.yhat.rf <- ifelse(smote.prob.rf >= 0.5, 1, 0) #prediction of test set outcome

caret::confusionMatrix(data=factor(smote.yhat.rf, levels=c(0,1)), reference=factor(y.test, levels=c(0,1)), positive='1') #eval metrics
smote.roc.rf <- pROC::roc(response=y.test, predictor=smote.prob.rf, levels=c(0,1), direction="<")
  plot(smote.roc.rf) #ROC specificity vs sensitivity
  print(smote.roc.rf) #AUC
  ci.auc(smote.roc.rf) #AUC CI

print(pROC::coords(smote.roc.rf, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity", "auc"), transpose=F)) #point metrics at threshold 0.5
print(pROC::ci.coords(smote.roc.rf, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics CI

rf.pred.smote <- ROCR::prediction(smote.prob.rf, y.test)
rf.roc.smote <- ROCR::performance(rf.pred.smote, "tpr", "fpr")
  rf.auc.smote <- performance(rf.pred.smote, "auc")

plot(rf.roc.smote) #ROC  fpr vs tpr
  print(rf.auc.smote@y.values[[1]]) #AUC


#SMOTE BART---------------------------------------------------------------------
smote.x.train <- smote.train[, 1:4] #train set predictors
smote.y.train <- smote.train$Outcome #train set response
x.test <- test[, 1:4] #test set predict

#set.seed(1)
smote.bart.fit <- pbart(smote.x.train, smote.y.train, x.test) #model

smote.order <- order(smote.bart.fit$varcount.mean, decreasing=T) #avg variable count in decreasing order
smote.bart.fit$varcount.mean[smote.order] #variable importance

smote.prob.bart <- smote.bart.fit$prob.test.mean #probability of test set outcome
smote.yhat.bart <- ifelse(smote.prob.bart >= 0.5, 1, 0) #prediction of test set outcome

caret::confusionMatrix(data=factor(smote.yhat.bart, levels=c(0,1)), reference=factor(y.test, levels=c(0,1)), positive='1') #eval metrics
smote.roc.bart <- pROC::roc(response=y.test, predictor=smote.prob.bart, levels=c(0,1), direction="<")
  plot(smote.roc.bart) #ROC specificity vs sensitivity
  print(smote.roc.bart) #AUC
  ci.auc(smote.roc.bart) #AUC CI

print(pROC::coords(smote.roc.bart, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity", "auc"), transpose=F)) #point metrics at threshold 0.5
print(pROC::ci.coords(smote.roc.bart, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics CI

bart.pred.smote <- ROCR::prediction(smote.prob.bart, y.test)
bart.roc.smote <- ROCR::performance(bart.pred.smote, "tpr", "fpr")
  bart.auc.smote <- performance(bart.pred.smote, "auc")

plot(bart.roc.smote) #ROC fpr vs tpr
  print(bart.auc.smote@y.values[[1]]) #AUC


#SMOTE NEURAL NETWORK-----------------------------------------------------------
#set.seed(1)
smote.nn <- neuralnet(formula=Outcome~AST.ALT+Blrb.Peak+Crtn.Peak+INR.Peak,
                      data=smote.train, hidden=2, rep=3, linear.output=FALSE, threshold=0.001)
plot(smote.nn, rep="best") #model

smote.prob.nn <- predict(smote.nn, newdata=test) #probability of test set outcome
smote.yhat.nn <- ifelse(smote.prob.nn >= 0.5, 1, 0) #prediction of test set outcome

caret::confusionMatrix(data=factor(smote.yhat.nn, levels=c(0,1)), reference=factor(y.test, levels=c(0,1)), positive='1') #eval metrics
smote.roc.nn <- pROC::roc(response=y.test, predictor=smote.prob.nn, levels=c(0,1), direction="<")
  plot(smote.roc.nn) #ROC specificity vs sensitivity
  print(smote.roc.nn) #AUC
  ci.auc(smote.roc.nn) #AUC CI

print(pROC::coords(smote.roc.nn, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity", "auc"), transpose=F)) #point metrics at threshold 0.5
print(pROC::ci.coords(smote.roc.nn, x=0.5, input='threshold', ret=c("accuracy", "sensitivity", "specificity"), transpose=F)) #point metrics CI

nn.pred.smote <- ROCR::prediction(smote.prob.nn, y.test)
nn.roc.smote <- ROCR::performance(nn.pred.smote, "tpr", "fpr")
  nn.auc.smote <- performance(nn.pred.smote, "auc")

plot(nn.roc.smote) #ROC fpr vs tpr
  print(nn.auc.smote@y.values[[1]]) #AUC


#SMOTE ROC CURVES---------------------------------------------------------------
plot(smote.roc.logreg, col="#F9AE9F", lty=1, lwd=5, family="serif", cex.lab=1.5, cex.axis=1.5)
plot(smote.roc.tree, col="#FADF57", lty=1, lwd=5, add=TRUE)
plot(smote.roc.rf, col="#AACD9D", lty=1, lwd=5, add=TRUE) 
plot(smote.roc.bart, col="#98C1F1", lty=1, lwd=5, add=TRUE)
plot(smote.roc.nn, col="gray", lty=1, lwd=5, add=TRUE) 
legend("bottomright", inset=0.025, cex=1.5, text.font=6, 
       legend=c("Logistic Regression", "Regression Tree", "Random Forest", "BART", "Neural Network"),
       col=c("#F9AE9F","#FADF57","#AACD9D","#98C1F1","gray"), lty=(1), lwd=(5))



















