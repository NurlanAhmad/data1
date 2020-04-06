library(tidyverse)
library(caret)
library(svm)
library(randomForest)
library(e1071)
library(xgboost)
library(DataExplorer)
library(janitor)
library(pROC)
library(doSNOW)


#--Activate parallel processing--------------------------
cl <- makeCluster(4, type="SOCK")
registerDoSNOW(cl)
stopCluster(cl)
#-------------------------------------------------------



#--Exploring data------------
df <- read_csv("bank-marketing-campaigns-dataset/data.csv")
view(df)
glimpse(df)
summary(df)
plot_missing(df)
plot_intro(df)

map_dbl(df,n_distinct)
df <- mutate_if(df,is.character,as.factor)
df<- rename(df,target=y)
str(df)


#--Preparing data for modelling-------------
dmy <- dummyVars("~ .",data=df[,c("job","marital","education","default","housing",
                                  "loan","contact","month","day_of_week","poutcome")])

dummy.data<-data.frame(predict(dmy,df[,c("job","marital","education","default","housing",
                             "loan","contact","month","day_of_week","poutcome")]))

mydata <- cbind(dummy.data,select_if(df,is.numeric),target=df$target)
scale2 <- function(x, na.rm = FALSE) (x - mean(x, na.rm = na.rm)) / sd(x, na.rm)

mydata<-mutate_at(mydata,c("age","duration","campaign",
                   "pdays","previous","emp.var.rate",
                   "cons.price.idx","euribor3m","nr.employed"),scale2)

#--KNN MODEL---------------
set.seed(11)
indx <- createDataPartition(mydata$target,p=0.7,list = F)
train <- mydata[indx,]
test <- mydata[-indx,]

set.seed(13)
Control.knn <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  allowParallel = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = FALSE)

set.seed(15)
knn_model <- train(target ~ .,
                      method = "knn",
                      data = train,
                      trControl=Control.knn,
                      metric="ROC",
                      tuneLength=10)

knn_model
plot(knn_model)
knn.predict <- predict(knn_model,test)
mean(knn.predict==test$target)
confusionMatrix(knn.predict,test$target)
knn.predict <- as.factor(knn.predict)
knn.roc <- roc(as.numeric(test$target),as.numeric(knn.predict))
plot(knn.roc, print.auc=TRUE)

#--PENALIZED LOGISTIC REGRESSION-----------
set.seed(17)
control.glmnet <- trainControl(method="cv", number=10,
                       classProbs=TRUE, summaryFunction=twoClassSummary)

set.seed(19)
glmnet_model <- train(target~.,data = train, method = "glmnet", 
                             trControl = control.glmnet,metric = "ROC",
                             tuneGrid = expand.grid(alpha = 0.3,
                                                    lambda = seq(0.001,0.1,by = 0.001)))
glmnet_model
glmnet_model$bestTune
glm.predict <-predict(glmnet_model,test)
mean(glm.predict == test$target)
confusionMatrix(glm.predict,test$target)
#--RANDOM FOREST-----------------

set.seed(21)
control.rf <-trainControl(method="cv", number=10,
                          classProbs=TRUE, summaryFunction=twoClassSummary)
set.seed(23)
rf_model <- train(target~.,data=train,method="rf",
                                        trControl=control.rf,
                                                 metric = "Accuracy",
                                                   importance = TRUE,
                                                            tuneLength = 10)
rf_model
plot(rf_model)
importance(model$finalModel)
varImpPlot(model$finalModel)

rf.predict <- predict(rf_model,test)
mean(rf.predict==test$target)
confusionMatrix(rf.predict,test$target)

#--SVM MODEL------------------
set.seed(25)
control.svm <- trainControl(method="cv",number = 10,
                            classProbs=TRUE)
svm_grid <-expand.grid(C=c(0.1,1,10,100,1000),
                       sigma=c(0.5, 1,2,3,4))
set.seed(27)
svm_model <- train(target~.,data=train,method="svmRadial",
                   metric="Accuracy",tuneGrid=svm_grid,trControl=control.svm)
svm_model
plot(svm_model)
svm.predict <- predict(svm_model,test$target)
mean(svm.predict==test$target)
confusionMatrix(svm.predict,test$target)


#--GBM MODEL-----------------
set.seed(29)
gbm_grid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
control.gbm <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 3,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(825)
gbm_model <- train(target ~ ., data = train, 
                 method = "gbm", 
                 distribution = "bernoulli",
                 trControl = fitControl, 
                 verbose = FALSE, 
                 tuneGrid = gbmGrid,
                 ## Specify which metric to optimize
                 metric = "ROC")
gbm_model
plot(gbm_model)
gbm.predict <- predict(gbm_model,test)
mean(gbm.predict==test$target)
confusionMatrix(gbm.predict, test$target)











