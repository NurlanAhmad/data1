library(tidyverse)
library(caret)
library(randomForest)
library(class)
library(e1071)
library(DataExplorer)
library(janitor)
library(cluster)
library(factoextra)


#LOAD DATA
train <- read_csv("titanic_train.csv")
test <- read_csv("titanic_test.csv")

#ADD "SURVIVED" VARIABLE TO THE TEST SET
test.survived <- tibble(survived=rep("None",nrow(test)))
test <- bind_cols(test.survived,test)

#COMBINE DATA SETS
data.combined <- rbind(train,test)


#FIND AND CHANGE DATA TYPES
str(train)
str(test)
str(data.combined)
apply(data.combined,2,n_distinct)

data.combined<-data.combined %>% mutate_if(is.character,as.factor)
data.combined$name <- as.character(data.combined$name)
data.combined$ticket <- as.character(data.combined$ticket)
data.combined$pclass <- as.factor(data.combined$pclass)
data.combined$sibsp <- as.factor(data.combined$sibsp)
data.combined$parch <- as.factor(data.combined$parch)

glimpse(data.combined)

#LOOK AT SURVIVAL RATES
tabyl(data.combined$survived)

#DISTURBUTION ACROSS CLASSES
tabyl(data.combined$pclass)


#VISUAL ANALYSIS OF DATA
plot_intro(data.combined)
plot_missing(data.combined) #for missing values
plot_bar(data.combined)
plot_histogram(data.combined)


ggplot(train,aes(x=as.factor(pclass),fill=as.factor(survived)))+geom_bar(position = "dodge")+xlab("Pclass")+
  ylab("Total count")+labs(fill="Survived")

ggplot(train,aes(x=as.factor(sex),fill=as.factor(survived)))+geom_bar(position = "dodge")+xlab("Sex")+
  ylab("Total count")+labs(fill="Survived")+theme_minimal()+scale_fill_brewer(palette = "Set2")



#HOW MANY UNIQE NAMES 
n_distinct(data.combined$name) #two duplicate names
dub.names <-data.combined[which(duplicated(data.combined$name)),"name",drop=T]
data.combined[which(data.combined$name %in% dub.names),]

#EXTARCT TITLES AND VISUALIZE 
data.combined <- data.combined%>%mutate(title=case_when(
  str_extract(data.combined$name,"Miss")=="Miss"~"Miss",
  str_extract(data.combined$name,"Mr")=="Mr"~"Mr",
  str_extract(data.combined$name,"Mrs")=="Mrs"~"Mrs",
  str_extract(data.combined$name,"Master")=="Master"~"Master",
  str_extract(data.combined$name,"Dr")=="Dr"~"Dr",
  str_extract(data.combined$name,"Col")=="Col"~"Col",
  str_extract(data.combined$name,"Rev")=="Rev"~"Rev",
  TRUE~"None"))

data.combined$title <- as.factor(data.combined$title)

ggplot(data.combined[1:891,],aes(x=title,fill=survived))+geom_bar()+facet_wrap(~pclass)+
  ggtitle("Pclass")+xlab("Title")+ylab("Total count")+labs(fill="Survived")+
  scale_fill_brewer(palette = "Set1")+theme_dark()



#DISTRIBUTION OF AGE BY SEX, AGE AND CLASS
summary(data.combined$age)
sum(is.na(data.combined$age))/nrow(data.combined)


ggplot(train,aes(x=age,fill=as.factor(survived)))+  
  facet_wrap(~sex+pclass)+geom_histogram(bindwith=10)+
  xlab("Age")+ylab("Total count")+theme_minimal() 


summary(data.combined[data.combined$title=="Master","age"])   #Boys
summary(data.combined[data.combined$title=="Miss","age"])     #Misses
summary(data.combined[data.combined$title=="Mr","age"])       #Men
summary(data.combined[data.combined$title=="Mrs","age"])      #Girls


ggplot(data.combined[data.combined$survived!="None"&data.combined$title=="Miss",],aes(x=age,fill=survived))+
  facet_wrap(~pclass)+geom_histogram(binwidth = 10)+
  ggtitle("Age distribution by class")+
  xlab("Age")+ylab("Total count")
 
ggplot(data.combined[data.combined$survived!="None"&data.combined$title=="Mr",],aes(x=age,fill=survived))+
  facet_wrap(~pclass)+geom_histogram(binwidth = 10)+
  ggtitle("Age distribution by class")+
  xlab("Age")+ylab("Total count")


ggplot(data.combined[data.combined$survived!="None"&data.combined$title=="Mrs",],aes(x=age,fill=survived))+
  facet_wrap(~pclass)+geom_histogram(binwidth = 10)+
  ggtitle("Age distribution by class")+
  xlab("Age")+ylab("Total count")

ggplot(data.combined[data.combined$survived!="None"&data.combined$title=="Master",],aes(x=age,fill=survived))+
  facet_wrap(~pclass)+geom_histogram(binwidth = 10)+
  ggtitle("Age distribution by class")+
  xlab("Age")+ylab("Total count")


#CREATING FAMILY SIZE
sibsp <- c(train$sibsp,test$sibsp)
parch <- c(train$parch,test$parch)
data.combined$family.size <- as.factor(sibsp+parch+1)
glimpse(data.combined)

ggplot(data.combined[1:891,],aes(x=family.size,fill=survived))+
  geom_bar()+
  facet_wrap(~pclass+title)+
  ggtitle("Pclass and Title")+
  xlab("Family size")+ylab("Total count")+ylim(0,300)+labs(fill="Survived")



#_____________________MODELLING______________________


#----------------RANDOM FOREST-----------------------------
rf.train <- data.combined[1:891,c("pclass","title","survived")]
set.seed(107)
rf1 <- randomForest(survived~.,data=rf.train,importance=T,ntree=1000)
rf1
varImpPlot(rf1)


rf.train2 <- data.combined[1:891,c("pclass","title","survived","sibsp")]
set.seed(101)
rf2 <-randomForest(survived~.,data=rf.train2,importance=T,ntree=1000)
rf2
varImpPlot(rf2)

rf.train3 <- data.combined[1:891,c("pclass","title","survived","parch","sibsp")]
set.seed(105)
rf3 <-randomForest(survived~.,data=rf.train3,importance=T,ntree=1000)
rf3
varImpPlot(rf3)


rf.train4 <- data.combined[1:891,c("pclass","title","survived","family.size")]
set.seed(109)
rf4 <-randomForest(survived~.,data=rf.train4,importance=T,ntree=1000)
rf4
varImpPlot(rf4)

#PERDICT AND SUBMIT
rf.test1 <- data.combined[892:1309,c("pclass","title","family.size")]
predict.rf <- predict(rf4,rf.test1)
table(predict.rf)
rf.submit <- data.frame(PassengerId=892:1309,Survived=predict.rf)
write_csv(rf.submit,"randomforestsubmit.csv")


#-------------------------CARET RANDOM FOREST------------

rf.train5 <- data.combined[1:891,c("pclass","title","survived","family.size")]
rf.test5 <- data.combined[892:1309,c("pclass","title","family.size")]

set.seed(133)
crf.model <-train(survived~.,data=rf.train5,method="rf",ntree=1000,tunelength=3,
                  trControl=trainControl("repeatedcv",number=5,repeats = 10),
                  importance=TRUE)
crf.model
crf.model$bestTune
crf.model$finalModel
importance(crf.model$finalModel)
varImp(crf.model)

predict.crf <- predict(crf.model,rf.test5)
table(predict.crf)
crf.submit <- data.frame(PassengerId=892:1309,Survived=predict.crf)
write_csv(crf.submit,"caretrandomforestsubmit.csv")

#--------------SVM------------------------------
mydata <-data.combined[1:891,c("pclass","title","survived","family.size")]
mytest<-data.combined[892:1309,c("pclass","title","family.size")]

set.seed(801)
svm1 <- tune(svm,survived~.,data=mydata,kernel="radial",
             ranges = list(cost = c(0.1,1,10,100,1000),
                           gamma = c(0.5,1,2,3,4)))

svm1$best.model
predict.svm <- predict(svm1$best.model,mytest)
table(predict.svm)

svm.submit <- data.frame(PassengerId=892:1309,Survived=predict.crf)
write_csv(svm.submit,"svmtitanic.csv")



































 















