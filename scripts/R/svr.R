library(e1071);

train <- read.csv('~/TFG/data/train_proc.csv');
test <- read.csv('~/TFG/data/test_proc.csv');
trainX <- train[,-81];
trainY <- train$SalePrice;

mod<- svm(SalePrice ~.,train)
pred <- predict(mod, test)

table<- data.frame (cbind(test[,1],pred))
colnames(table)[1]<-'Id'
colnames(table)[2]<-'SalePrice'

write.csv(table[,1:2],file=paste('~/TFG/scripts/predictions/prediction',Sys.time(),'csv',sep = '.')
          ,row.names=FALSE)
