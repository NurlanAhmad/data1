library(corrplot)
library(ggplot2)
library(randomForest)
library(dplyr)
library(corrplot)
library(caret)



test <- read_csv("house-prices-advanced-regression-techniques/test.csv")
train <- read_csv("house-prices-advanced-regression-techniques/train.csv")



fulldt <- rbind(train[,-81], raw.test)
fulldt <- cbind(fulldt, Set = c(rep("Train", times = dim(train)[1]),
                                rep("Test", times = dim(test)[1])))

x <- colSums(sapply(fulldt, is.na))

# Set table
x <- data.frame(Variables = names(x), NA.Count = x); rownames(x) <- c()

# Remove variables that don't have missing values
x <- x %>% filter(NA.Count > 0)


y <- c("LotFrontage", "MasVnrArea", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath")
fulldt[,y] <- apply(fulldt[,y], 2, 
                    function(x) {
                      replace(x, is.na(x), 0)
                    }
)

y <- c("Alley", "BsmtQual", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtCond")
fulldt[,y] <- apply(fulldt[,y], 2, 
                    function(x) {
                      replace(x, is.na(x), "None")
                    }
)



y <- c("MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "MasVnrType", "Electrical", "KitchenQual", "Functional", "SaleType")
fulldt[,y] <- apply(fulldt[,y], 2, 
                    function(x) {
                      replace(x, is.na(x), names(which.max(table(x))))
                    }
)


y <- c("GarageCars", "GarageArea", "BsmtFinSF1")
fulldt[,y] <- apply(fulldt[,y], 2, 
                    function(x) {
                      replace(x, is.na(x), median(x, na.rm = T))
                    }
)



fulldt$GarageYrBlt[is.na(fulldt$GarageYrBlt)] <- fulldt$YearBuilt[is.na(fulldt$GarageYrBlt)]



# Colect name of variables that are character
class.list <- sapply(fulldt, class)
class.list.character <- names(class.list[which(class.list=="character")])

# Convert to factor
fulldt[class.list.character] <- lapply(fulldt[class.list.character], factor)

# Fix MSSubClass class
fulldt$MSSubClass <- factor(fulldt$MSSubClass)



# Create a "total area" feature by adding the basement area and ground living area
fulldt$TotalArea <- fulldt$GrLivArea + fulldt$TotalBsmtSF

# Create a "total number of baths" feature by adding all bathroom features
fulldt$TotalBaths <- fulldt$BsmtFullBath + 
  fulldt$BsmtHalfBath +
  fulldt$FullBath + 
  fulldt$HalfBath

# Create a "area aboveground" feature by adding the areas of the first and second floor
fulldt$AreaAbvground <- fulldt$`1stFlrSF` + fulldt$`2ndFlrSF`

# Subset numerical variables that are from the "train" set
fulldt.num.train <- fulldt %>% filter(Set == "Train") %>% 
  select(which(sapply(.,is.integer)), which(sapply(., is.numeric))) %>%
  mutate(SalePrice = raw.train$SalePrice) #Add the "SalePrice" variable


correlation <- round(cor(fulldt.num.train),2)

corrplot(correlation, method = "circle")

# Set a table with "SalePrice" correlation
x <- data.frame(Variables = rownames(correlation), 
                Cor = correlation[, "SalePrice"])

# Order it by correlation
x <- x[order(x$Cor, decreasing = T),]

# Pick only values that have strong positive and negative correlation
x <- x[which(x$Cor > 0.5 | x$Cor < -0.5),]
rownames(x) <- c()



# Subset numerical variables that are from the "test" set
fulldt.fac.train <- fulldt %>% filter(Set == "Train") %>%
  select(Id, which(sapply(., is.factor))) %>%
  mutate(SalePrice = raw.train$SalePrice) # Add SalePrice variable

fulldt.fac.test <- fulldt %>% filter(Set == "Test") %>%
  select(Id, which(sapply(., is.factor)))

# Run RF algorithm will all factor variables
rf <- randomForest(SalePrice ~ ., data = fulldt.fac.train, importance = T)
# Create Table with importance values
importance.table <- data.frame(Names = rownames(importance(rf)), '%IncMSE' = importance(rf)[,1])

# Order table
importance.table <- importance.table[order(importance.table[,2], decreasing = T),]
rownames(importance.table) <- c()


fulldt.num.train$SalePrice <- log(fulldt.num.train$SalePrice)
fulldt.fac.train$SalePrice <- log(fulldt.fac.train$SalePrice)

# Subset the train rows and selected features
dt.train <- fulldt %>% filter(Set == "Train") %>%
  select("Id", "OverallQual", "TotalArea", "AreaAbvground", "GarageArea", "TotalBaths", "YearBuilt", 
         "Neighborhood", "MSSubClass", "FireplaceQu", "ExterQual", "KitchenQual", "BsmtQual", "HouseStyle") %>%
  mutate(SalePrice = log(raw.train$SalePrice)) # Don't forget to do the log transformation

# Same for the test features
dt.test <- fulldt %>% filter(Set == "Test") %>%
  select("Id", "OverallQual", "TotalArea", "AreaAbvground", "GarageArea", "TotalBaths", "YearBuilt", 
         "Neighborhood", "MSSubClass", "FireplaceQu", "ExterQual", "KitchenQual", "BsmtQual", "HouseStyle")

# Random Forest model
fit <- randomForest(SalePrice ~ ., data = dt.train, importance = T)

# Use new model to predict SalePrice values from the test set
pred <- exp(predict(fit , newdata = dt.test))

# Export Result
write.csv(x = data.frame(Id = raw.test$Id, SalePrice = pred), row.names = F, file = "./submission.csv")