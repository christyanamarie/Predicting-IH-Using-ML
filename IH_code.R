#Code for Predicting the Outcome of Ischemic Hepatitis with Real Patient Data Using Machine Learning Tools

#LIBRARIES----------------------------------------------------------------------
#libraries needed for functions used
library(readxl) #read_excel
library(caret) #createDataPartion, confusionMatrix
library(tree) #tree
library(BART) #pbart
library(neuralnet) #neuralnet
library(smotefamily) #SMOTE
library(pROC) #roc
library(ggplot2) #ggplot


#DATA---------------------------------------------------------------------------
data <- read_excel("IH_cleaned_data.xlsx") #read in data file

#AST/ALT------------------------------------------------------------------------
r <- cor(data$ALT_Peak, data$AST_Peak) #correlation coefficient between ALT & AST
AST.ALT <- data$AST_Peak/data$ALT_Peak
vars <- data.frame(AST.ALT, data$Blrb_Peak, data$Crtn_Peak, data$INR_Peak)

#SCALE--------------------------------------------------------------------------
data.scaled <- as.data.frame(lapply(vars, scale)) #normalize predictors

df <- cbind(data.scaled, data[,6]) #combine normalized vars with response var
colnames(df)[5] <-  "Outcome" #name response variable
colnames(df)[4] <-  "INR.Peak"
colnames(df)[3] <-  "Crtn.Peak"
colnames(df)[2] <-  "Blrb.Peak"


#SUBSETS------------------------------------------------------------------------
set.seed(1)
index <- createDataPartition(unlist(df[,5]), p=0.8, list=FALSE, times=1)
train <- df[index,] #training set (80%)
test <- df[-index,] #testing set (20% validation)

y.test <- test$Outcome #test set response


#LOGISTIC REGRESSION------------------------------------------------------------
full.model <- glm(Outcome~., data=train, family="binomial") #full logistic model
summary(full.model) 

null.model <- glm(Outcome~1, data=train, family="binomial") #empty logistic model
logreg <- step(null.model, scope=list(upper=full.model), direction="both", test="Chisq", trace=F) #model selection
summary(logreg) #stepwise model

prob.logreg <- predict(logreg, newdata=test, type="response") #probability of test set outcome
yhat.logreg <- round(prob.logreg) #prediction of test set outcome

confusionMatrix(factor(yhat.logreg), factor(y.test)) #eval metrics
roc.logreg <- roc(y.test, prob.logreg, plot=TRUE, print.auc=TRUE, print.auc.y=0.15) #ROC & AUC


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
yhat.tree <- round(prob.tree) #prediction of test set outcome

confusionMatrix(data=factor(yhat.tree), reference=factor(y.test)) #eval metrics
roc.tree <- roc(y.test, prob.tree, plot=TRUE, print.auc=TRUE, print.auc.y=0.15) #ROC & AUC


#BART---------------------------------------------------------------------------
x.train <- train[, 1:4] #train set predictors
y.train <- train$Outcome #train set response
x.test <- test[, 1:4] #test set predict

#set.seed(1)
bart.fit <- pbart(x.train, y.train, x.test) #model

prob.bart <- bart.fit$prob.test.mean #probability of test set outcome
yhat.bart <- round(prob.bart) #prediction of test set outcome

order <- order(bart.fit$varcount.mean, decreasing=T) #avg variable count in decreasing order
bart.fit$varcount.mean[order] #variable importance

confusionMatrix(data=factor(yhat.bart), reference=factor(y.test)) #eval metrics
roc.bart <- roc(y.test, prob.bart, plot=TRUE, print.auc=TRUE, print.auc.y=0.15) #ROC & AUC


#NEURAL NETWORK-----------------------------------------------------------------
set.seed(5)
nn <- neuralnet(Outcome~Crtn_Peak+INR_Peak+ALT_Peak+AST_Peak+Blrb_Peak,
                data=train, hidden=c(3,3,3), rep=3, linear.output=FALSE, threshold=0.001)
plot(nn, rep="best") #model

prob.nn <- predict(nn, newdata=test) #probability of test set outcome
yhat.nn <- round(prob.nn) #prediction of test set outcome

confusionMatrix(data=factor(yhat.nn), reference=factor(y.test)) #eval metrics
roc.nn <- roc(y.test, prob.nn, plot=TRUE, print.auc=TRUE, print.auc.y=0.15) #ROC & AUC


#ROC CURVES---------------------------------------------------------------------
plot(roc.logreg, col="#F9AE9F", lty=1, lwd=5, family="serif", cex.lab=1.5, cex.axis=1.5)
plot(roc.tree, col="#FADF57", lty=1, lwd=5, add=TRUE)
plot(roc.bart, col="#AACD9D", lty=1, lwd=5, add=TRUE)
plot(roc.nn, col="#98C1F1", lty=1, lwd=5, add=TRUE) 
legend("bottomright", cex=1.5, text.font=6, 
       legend=c("Logistic Regression", "Regression Tree", "BART", "Neural Network"),
       col=c("#F9AE9F","#FADF57","#AACD9D","#98C1F1"), lty=(1), lwd=(5))


#SMOTE--------------------------------------------------------------------------
prop.table(table(data$Outcome)) #original response proportion

n1 <- table(data$Outcome)['1'] #number of "1" responses
n0 <- table(data$Outcome)['0'] #number of "0" responses
r0 <- 0.50 #ideal proportion of responses
ntimes <- ((1-r0)/r0)*(n1/n0)-1 #formula

data[7] <- 1:length(data$Outcome) #add index to original data
colnames(data)[7] <-  "Index" #name index for original data

set.seed(1)
SMOTE <- SMOTE(X=data, target=data$Outcome, K=5, dup_size=ntimes) #apply algorithm
SMOTE.DATA <- SMOTE$data #new dataset
SMOTE.data <- SMOTE.DATA[,-8] #remove duplicated class
  #order by original data index to easily identify synthetic observations  
  SMOTE.data <- SMOTE.data[order(unlist(SMOTE.data[,7])),] 
smote.data <- SMOTE.data[,-7] #remove index for completeness

prop.table(table(smote.data$Outcome)) #SMOTE response proportion

#plot data responses
ggplot(data, aes(x=ALT_Peak, y=AST_Peak, color=factor(Outcome))) +
  geom_point() + scale_color_manual(values=c('#98C1F1','#F9AE9F')) +
  ggtitle("Data") +  theme(plot.title=element_text(hjust=0.5))
#plot SMOTE data responses (can see increase in minority "0")
ggplot(smote.data, aes(x=ALT_Peak, y=AST_Peak, color=factor(Outcome))) +
  geom_point() +scale_color_manual(values=c('#98C1F1','#F9AE9F')) +
  ggtitle("SMOTE Data") +  theme(plot.title=element_text(hjust=0.5))

r <- cor(data$ALT_Peak, data$AST_Peak) #correlation coefficient between ALT & AST


#SMOTE SCALE--------------------------------------------------------------------
smote.data.scaled <- as.data.frame(lapply(smote.data[,1:5], scale)) #normalize predictors
smote.df <- cbind(smote.data.scaled, smote.data[,6]) #combine normalized vars with response var
colnames(smote.df)[6] <- "Outcome" #name response variable


#SMOTE SUBSETS------------------------------------------------------------------
set.seed(1)
smote.index <- createDataPartition(unlist(smote.df[,6]), p=0.8, list=FALSE, times=1)
smote.train <- smote.df[smote.index,] #training set (80%)
smote.test <- smote.df[-smote.index,] #testing set (20% validation)

smote.y.test <- smote.test$Outcome #test set response


#SMOTE LOGISTIC REGRESSION------------------------------------------------------
smote.full.model <- glm(Outcome~., data=smote.train, family="binomial") #full logistic model
summary(smote.full.model)

set.seed(1)
smote.null.model <- glm(Outcome~1, data=smote.train, family="binomial") #empty logistic model
smote.logreg <- step(smote.null.model, scope=list(upper=smote.full.model), direction="both", test="Chisq", trace=F) #model selection
summary(smote.logreg) #stepwise model

smote.prob.logreg <- predict(smote.logreg, newdata=smote.test, type="response") #probability of test set response
smote.yhat.logreg <- round(smote.prob.logreg) #prediction of test set response

confusionMatrix(factor(smote.yhat.logreg), factor(smote.y.test)) #eval metrics
smote.roc.logreg <- roc(smote.y.test, smote.prob.logreg, plot=TRUE, print.auc=TRUE, print.auc.y=0.15) #ROC & AUC


#SMOTE REGRESSION TREE----------------------------------------------------------
smote.full.tree <- tree(Outcome~., data=smote.train) #full regression tree
plot(smote.full.tree) #plot full regression tree
text(smote.full.tree, pretty=0) #add labels to plot

set.seed(5)
cv <- cv.tree(smote.full.tree, K=5) #5-fold cross-validation
plot(cv$size, cv$dev, type='b') #determine "best" number of nodes for pruning (want: min)

smote.tree <- prune.tree(smote.full.tree, best=5) #pruned regression tree
plot(smote.tree) #plot pruned regression tree
text(smote.tree, pretty=0) #add labels to plot

smote.prob.tree <- predict(smote.tree, newdata=smote.test) #probability of test set outcome
smote.yhat.tree <- round(smote.prob.tree) #prediction of test set outcome

confusionMatrix(data=factor(smote.yhat.tree), reference=factor(smote.y.test)) #eval metrics
smote.roc.tree <- roc(smote.y.test, smote.prob.tree, plot=TRUE, print.auc=TRUE, print.auc.y=0.15) #ROC & AUC


#SMOTE BART---------------------------------------------------------------------
smote.x.train <- smote.train[, 1:5] #train set predictors
smote.y.train <- smote.train$Outcome #train set response
smote.x.test <- smote.test[, 1:5] #test set predict

set.seed(1)
smote.bart.fit <- pbart(smote.x.train, smote.y.train, smote.x.test) #model

smote.prob.bart <- smote.bart.fit$prob.test.mean #probability of test set outcome
smote.yhat.bart <- round(smote.prob.bart) #prediction of test set outcome

smote.order <- order(smote.bart.fit$varcount.mean, decreasing=T) #avg variable count in decreasing order
smote.bart.fit$varcount.mean[smote.order] #variable importance

confusionMatrix(data=factor(smote.yhat.bart), reference=factor(smote.y.test)) #eval metrics
smote.roc.bart <- roc(smote.y.test, smote.prob.bart, plot=TRUE, print.auc=TRUE, print.auc.y=0.15) #ROC & AUC


#SMOTE NEURAL NETWORK-----------------------------------------------------------
set.seed(5)
smote.nn <- neuralnet(formula=Outcome~Crtn_Peak+INR_Peak+ALT_Peak+AST_Peak+Blrb_Peak,
                      data=smote.train, hidden=c(5,1), rep=3, linear.output=FALSE, threshold=0.001)
plot(smote.nn, rep="best") #model

smote.prob.nn <- predict(smote.nn, newdata=smote.test) #probability of test set outcome
smote.yhat.nn <- round(smote.prob.nn) #prediction of test set outcome

confusionMatrix(data=factor(smote.yhat.nn), reference=factor(smote.y.test)) #eval metrics
smote.roc.nn <- roc(smote.y.test, smote.prob.nn, plot=TRUE, print.auc=TRUE, print.auc.y=0.15) #ROC & AUC


#SMOTE ROC CURVES---------------------------------------------------------------
plot(smote.roc.logreg, col="#F9AE9F", lty=1, lwd=5, family="serif", cex.lab=1.5, cex.axis=1.5)
plot(smote.roc.tree, col="#FADF57", lty=1, lwd=5, add=TRUE)
plot(smote.roc.bart, col="#AACD9D", lty=1, lwd=5, add=TRUE)
plot(smote.roc.nn, col="#98C1F1", lty=1, lwd=5, add=TRUE) 
legend("bottomright", cex=1.5, text.font=6, 
       legend=c("Logistic Regression", "Regression Tree", "BART", "Neural Network"),
       col=c("#F9AE9F","#FADF57","#AACD9D","#98C1F1"), lty=(1), lwd=(5))






