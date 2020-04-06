library(caret)
library(gbm)
library(plyr)
library(randomForest)
library(janitor)
library(e1071)
library(tidyverse)
library(doSNOW)
library(DataExplorer)
library(corrr)
library(pROC)
library(broom)
library(xgboost)



#----------------
cl <- makeCluster(4, type="SOCK")
registerDoSNOW(cl)
stopCluster(cl)
#----------------


df <- read_csv("creditcardfraud/creditcard.csv")
View(df)
summary(df)
nrow(df)
colnames(df)
df$Class<- as.factor(df$Class)
levels(df$Class) <- c("Not_Fraud", "Fraud")
tabyl(df$Class)
df$Time <- NULL
df$Amount<-scale(df$Amount)

fraud <- filter(df,Class=="Fraud")
not_fraud <- filter(df,Class=="Not_Fraud")%>%sample_n(40000)
df<-rbind(fraud,not_fraud)
write_csv(df,"creditcardfraud/data")


#------------------------------

plot_intro(df)
plot_boxplot(df,by="Class")
plot_density(df[,-31])
plot_correlation(df,type="continuous",title = "Correlation between variables")


ggplot(df, aes(x = Class, y = Amount)) + geom_boxplot() + 
  labs(x = 'Class', y = 'Amount') +
  ggtitle("Distribution of transaction amount by class")+theme_minimal()


#---------------------------------
set.seed(11)
indx1<-createDataPartition(df$Class,p=0.7,list=F)
train<-df[indx1,]
test<-df[-indx1,]

#------------LOGISTIC REGRESSION

model.glm <- glm(Class~.,data=train,family = "binomial")
summary(model.glm)
plot(model.glm)

model.data <- augment(model.glm) %>% mutate(index=1:n()) #extract model result and find outliers
model.data <- top_n(3,.cooksd)

ggplot(model.data,aes(index,.std.resid))+
         geom_point(aes(color=Class),alpha=0.5)+theme_bw() # plot the residuals

car::vif(model.glm) #find multicollinearity

glm.prob <- predict(model.glm,test,type="response")
contrasts(test$Class)
predicted1<-ifelse(glm.prob>0.5,"Fraud","Not_Fraud")
predicted1 <- as.factor(predicted1)
table(predicted1,test$Class)
mean(predicted1==test$Class)
confusionMatrix(predicted1,test$Class)
glm.roc<-roc(as.numeric(test$Class),as.numeric(predicted1))
plot.roc(glm.roc,print.auc = T)

#--------------Random Forest Caret
set.seed(3)
cvCtrl1 = trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                      classProbs = TRUE, summaryFunction = twoClassSummary)
newGrid1 = expand.grid(mtry = c(5,10,15,29))
classifierRandomForest <- train(Class ~ ., data = train, 
                                 trControl = cvCtrl1, 
                                 method = "rf", metric="ROC", tuneGrid = newGrid1)   
print(classifierRandomForest)


pred_rf <- as.data.frame(predict(classifierRandomForest$finalModel, test, type = "prob"))
pred_rf <- ifelse(pred_rf$Not_Fraud>pred_rf$Fraud,"Not_Fraud","Fraud")
pred_rf <- as.factor(pred_rf)
mean(pred_rf==test$Class)
rfroc = roc(test$Class, pred_rf)
plot(efroc, print.thres = "best")

threshold = coords(rfroc,x="best",best.method = "closest.topleft")[[1]] #get optimal cutoff threshold
predCut = factor( ifelse(predRoc[, "Yes"] > threshold, "Yes", "No") )



#----------------------GBM Caret

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), #maximum depth of each tree
                        n.trees = (1:30)*50, #total number of trees to fit
                        shrinkage = c(0.001,0.005,0.01,0.05,0.1),
                        n.minobsinnode = 100) #minimum number of observations in the terminal nodes

set.seed(13)

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(825)
gbmFit <- train(Class ~ ., data = train, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 tuneGrid = gbmGrid,
                 ## Specify which metric to optimize
                 metric = "ROC")
print(fitControl)


trellis.par.set(caretTheme())
plot(gbmFit)  

pred_gbm <- predict(gbmFit,test)
mean(pred_gbm==test$Class)


#-------------------------SVM Caret

set.seed(23)
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)


svmFit <- train(Class ~ ., data = training, 
                method = "svmRadial", 
                trControl = fitControl, 
                preProc = c("center", "scale"),
                tuneLength = 8,
                metric = "ROC")

plot(svmFit)
svmFit$bestTune

pred_svm <- predict(svmFit,test$Class)
mean(svmFit==test$Class)



set.seed(33)

adaptControl <- trainControl(method = "adaptive_cv",
                             number = 10, repeats = 10,
                             adaptive = list(min = 5, alpha = 0.05, 
                                             method = "gls", complete = TRUE),
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary,
                             search = "random")

set.seed(825)
svmAdapt <- train(Class ~ ., data = train,
                  method = "svmRadial", 
                  trControl = adaptControl, 
                  preProc = c("center", "scale"),
                  metric = "ROC",
                  tuneLength = 15)



plot(svmAdapt)
svmAdapt$bestTune

pred_svmAdapt <- predict(svmAdapt,test$Class)
mean(svmAdapt,test$Class)

#--------------------------XGBOOST------------------
set.seed(31)
indx1<-createDataPartition(df$Class,p=0.7,list=F)
train<-df[indx1,]
test<-df[-indx1,]


train.data<-train[,-30]
train.label<-train[,30]
train.label<-as.integer(unlist(train.label))-1

test.data<-test[,-30]
test.label<-test[,30]
test.label<-as.integer(unlist(test.label))-1

# Transform the two data sets into xgb.Matrix
xgb.train = xgb.DMatrix(data=as.matrix(train.data),label=train.label)
xgb.test = xgb.DMatrix(data=as.matrix(test.data),label=test.label)


# Define the parameters for multinomial classification
num_class = length(levels(df$Class))
params = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.7,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class
)

# Train the XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=10000,
  nthreads=4,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train,val2=xgb.test),
  verbose=2
)
# Review the final model and results
xgb.fit

# Predict outcomes with the test data
xgb.pred = predict(xgb.fit,as.matrix(test.data),reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = levels(df$Class)


# Use the predicted label with the highest probability
xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label = levels(df$Class)[test.label+1]

# Calculate the final accuracy
result = sum(xgb.pred$prediction==xgb.pred$label)/nrow(xgb.pred)
print(paste("Final Accuracy =",sprintf("%1.2f%%", 100*result)))
























































