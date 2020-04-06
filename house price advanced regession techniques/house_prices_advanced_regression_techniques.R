library(tidyverse)
library(DataExplorer)
library(caret)
library(randomForest)
library(doSNOW)
library(xgboost)
library(janitor)
library(Boruta)
library(corrplot)
library(plyr)

#Parallel processing
#----------------------------
cl <- makeCluster(4, type="SOCK")
registerDoSNOW(cl)
stopCluster(cl)
#---------------------------

#Load data
data.train <- read_csv("house-prices-advanced-regression-techniques/train.csv")
data.test <- read_csv("house-prices-advanced-regression-techniques/test.csv")

#Explore data
str(data.train)
summary(data.train)
summary
apply(data.train,2,n_distinct)
colnames(data.train)
colnames(data.test)
anyNA(data.train)

#Add missing variable to test data
data.test$SalePrice <- rep(0,nrow(data.test))

#Combine test and train
mydata <- rbind(data.trin,data.test)



#Clean and change data types
mydata<-rename(mydata,SnnPorch=`3SsnPorch`)
mydata[,c("Alley","MiscFeature","PoolQC","Fence","Id","FireplaceQu")] <- list(NULL)
round(colSums(is.na(mydata))/nrow(mydata),2)

chr<-as.data.frame(select_if(mydata,is.character)%>%map(~replace_na(.x,"None")))
colSums(is.na(chr))/nrow(chr)*100
summary(chr)
chr<-mutate_if(chr,is.character,as.factor)

levels(mydata$OverallCond)<-c("Very Excellent","Excellent","Very Good",
                              "Good","Above Average",	"Average","Below Average","Fair","Poor","Very Poor")
mydata$OverallQual<-as.factor(mydata$OverallQual)
levels(mydata$OverallQual)<-c("Very Excellent","Excellent","Very Good",
                              "Good","Above Average",	"Average","Below Average","Fair","Poor","Very Poor")

#Turn factors into dummy variables
dum <- dummyVars("~.",data=chr)
dum.chr <- as.data.frame(predict(dum,chr))
head(dum.chr)
str(dum.chr)


numbr<-as.data.frame(select_if(mydata,is.numeric)%>%map(~replace_na(.x,0)))
corrplot(cor(numbr),method = "circle",type = "upper")

numbr$OverallCond <-NULL
numbr$OverallQual<- NULL
numbr$TotalBaths <- numbr$BsmtFullBath + numbr$BsmtHalfBath + 
                                        numbr$FullBath + numbr$HalfBath

numbr$TotalArea1st2nd <- numbr$X1stFlrSF + numbr$X2ndFlrSF
numbr$TotalArea <- numbr$GrLivArea + numbr$TotalBsmtSF
numbr$MSSubClass<-NULL
numbr$GarageCars<-NULL
numbr$MoSold<-NULL
numbr$Age<-abs(numbr$YearBuilt-max(numbr$YearBuilt))
numbr$YearBuilt<-NULL
numbr$YrSold<-NULL
numbr$GarageYrBlt<-NULL
numbr$YearRemodAdd<-max(numbr$YearRemodAdd)-numbr$YearRemodAdd
numbr$TotalPorch<-numbr$OpenPorchSF+numbr$SnnPorch+
                           numbr$ScreenPorch+numbr$EnclosedPorch
plot_histogram(numbr)


df <- cbind(dum.chr,numbr)
colnames(df)
df<-clean_names(df)
df.train<-filter(df,sale_price!=0)
df.test<-filter(df,sale_price==0)
df.test$SalePrice<-NULL
df.train$sale_price<-log(df.train$sale_price)


#Find important variables
rf_model<-randomForest(sale_price~.,data=df.train,importance =T)
varImpPlot(rf_model)
br_model <- Boruta(sale_price~.,df.train,doTrace=2)
print(br_model)
signif <- names(br_model$finalDecision[br_model$finalDecision %in% c("Confirmed","Tentative")])
print(signif)
plot(br_model,cex.axis=.7,las=2)


br.train <- cbind(df.train[,c(signif,"sale_price")])
br.test <- cbind(df.test[,signif])


#XGBOOST model

set.seed(23)
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 10,number = 5)

xgb.grid<-expand.grid(nrounds = 1000,
            max_depth = c(2:10),
            eta = c(0.001,0.005,0.01,0.05),
            gamma = c(0.0, 0.2, 1),
            colsample_bytree = .7,
            min_child_weight = 3,
            subsample = c(.8, 1))

xgb_tune <-train(sale_price ~.,
                 data=br.train,
                 method="xgbTree",
                 metric = "RMSE",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid)

xgb_tune$finalModel

prediction <- data.frame(SalePrice=exp(predict(xgb_tune,br.test)))
write_csv(prediction,"house-prices-advanced-regression-techniques/pred_xgbtree2")












