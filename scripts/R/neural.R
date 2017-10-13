library(neuralnet)
# load data
train <- read.csv('~/TFG/data/train_proc.csv');
test <- read.csv('~/TFG/data/test_proc.csv');

maxs <- apply(train, 2, max) 
mins <- apply(train, 2, min)
train_ <- as.data.frame(scale(train, center = mins, scale = maxs - mins))

maxs <- apply(test, 2, max) 
mins <- apply(test, 2, min)
test_ <- as.data.frame(scale(test, center = mins, scale = maxs - mins))

# fit model
n <- names(train_[,-10])
f <- as.formula(paste("SalePrice ~", paste(n[!n %in% "SalePrice"], collapse = " + ")))
nn <- neuralnet(f,data=train_[,-10],hidden=c(5,3),linear.output=T)

# make predictions
pr.nn <- compute(nn,test_[,-10])
pr.nn_ <- pr.nn$net.result*(max(train$SalePrice)-min(train$SalePrice))+min(train$SalePrice)


table<- data.frame (cbind(test[,1],pr.nn_))
colnames(table)[1]<-'Id'
colnames(table)[2]<-'SalePrice'
write.csv(table[,1:2],file=paste('~/TFG/predictions/predictionNN',Sys.time(),'csv',sep = '.')
          ,row.names=FALSE)
