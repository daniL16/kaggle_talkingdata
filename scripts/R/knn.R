library(caret);

train <- read.csv('~/TFG/data/train_proc.csv');
test <- read.csv('~/TFG/data/test_proc.csv');
trainX <- train[,-81];
trainY <- train$SalePrice;

sample <- sample.int(n = nrow(train), size = floor(.75*nrow(train)), replace = F)
train_split <- train[sample, ]
test_split  <- trainX[-sample, ]
mod<-knnregTrain(train_split[,-81],test_split,train_split[,81])
table<- data.frame (cbind(test[,1],mod))
colnames(table)[1]<-'Id'
colnames(table)[2]<-'SalePrice'

write.csv(table[,1:2],file=paste('~/TFG/scripts/predictions/prediction',Sys.time(),'csv',sep = '.')
          ,row.names=FALSE)
