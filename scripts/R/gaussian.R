library(gptk)
library(caret)

# load data
train <- read.csv('~/TFG/data/train_proc.csv');
test <- read.csv('~/TFG/data/test_proc.csv');

trainX <- train[,-81];
trainY <- train$SalePrice;
trainY <- matrix(unlist(trainY), ncol = 1, byrow = TRUE)

options = gpOptions()
options$kern$comp = list("rbf", "white")

# fit model
mod <- gpCreate(q =80,d = 1,X  = trainX,y = trainY,options)
pred<- modelOut(mod,test)

# make predictions


table<- data.frame (cbind(test[,1],pred))
colnames(table)[1]<-'Id'
colnames(table)[2]<-'SalePrice'

write.csv(table[,1:2],file=paste('~/TFG/predictions/predictionGauss',Sys.time(),'csv',sep = '.')
          ,row.names=FALSE)

