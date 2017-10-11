library(GPfit)
library(caret)

# load data
train <- read.csv('~/TFG/data/train_proc.csv');
test <- read.csv('~/TFG/data/test_proc.csv');

preprocessParams <- preProcess(train, method=c("range"))
train <- predict(preprocessParams, train)

trainX <- train[,-81];
trainY <- train$SalePrice;

preprocessParams <- preProcess(test, method=c("range"))
test <- predict(preprocessParams, test)

# fit model
mod <- GP_fit(trainX,trainY)

# make predictions
pred <- predict(mod, test)

table<- data.frame (cbind(test[,1],pred))
colnames(table)[1]<-'Id'
colnames(table)[2]<-'SalePrice'

write.csv(table[,1:2],file=paste('~/TFG/scripts/predictions/predictionGauss',Sys.time(),'csv',sep = '.')
          ,row.names=FALSE)

