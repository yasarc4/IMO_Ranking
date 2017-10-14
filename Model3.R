setwd('/home/yasar/sling_media/')

train1 <- read.csv('data/transformed_data_val.csv')
train2 <- read.csv('data/transformed_data.csv')
train <- merge(train1,train2,by=c('X','Country.Name','Mean.Rank'))
library(DMwR)

train_rows <- sample(1:101,70)
train_x <- train[train_rows,3:6345]
test_x <- train[-train_rows,3:6345]
x_cols <- colnames(train_x)[2:6343]
library(corrplot)
correlations <-cor(train_x)
correlations_df <- as.data.frame(correlations)

library(randomForest)
rf_model <- randomForest(train_x[x_cols],train_x$Mean.Rank)
predicted_rf <- predict(rf_model, test_x)
stats::cor(predicted_rf,test_x$Mean.Rank)
## We get 79% correlation
regr.eval(predicted_rf,test_x$Mean.Rank)
# The MAPE metric is 33%
# RMSE if 23.9
feature_importance <- as.data.frame(rf_model$importance)
feature_importance$column_name <- rownames(feature_importance)
feature_importance$correlation_value <- sapply(feature_importance$column_name, FUN=function(x) correlations_df['Mean.Rank',x])
feature_importance <- feature_importance[with(feature_importance, order(-IncNodePurity,-correlation_value)),]

correlations_df <- as.data.frame(correlations)
top_features_correlations <- subset(correlations_df, select = feature_importance$column_name[1:20], rownames(correlations_df) %in% feature_importance$column_name[1:20])

corrplot.mixed(as.matrix(top_features_correlations))

library(h2o)
h2o.init()

train_hex <- as.h2o(train_x,'train_hex')
test_hex <- as.h2o(test_x, 'test_hex')
x_cols <- colnames(train_hex)[1:3171]
dl_model <- h2o.deeplearning(x_cols,'Mean.Rank',train_hex)

predicted <- h2o.predict(dl_model,test_hex[x_cols])

predicted$predict
test_x$Mean.Rank
stats::cor(as.data.frame(predicted)$predict,test_x$Mean.Rank)
## We get 31% correlation
regr.eval(as.data.frame(predicted)$predict,test_x$Mean.Rank)
# The MAPE metric is 36%
# RMSE if 36.1


library(catboost)
train_pool <- catboost.load_pool(data = train_x[x_cols],label = train_x$Mean.Rank)
test_pool <- catboost.load_pool(data=test_x[x_cols])
cat_model <- catboost.train(train_pool)
predicted_cat <- catboost.predict(cat_model,test_pool)

stats::cor(predicted_cat,test_x$Mean.Rank)
## We get 79% correlation
regr.eval(predicted_cat,test_x$Mean.Rank)
# The MAPE metric is 32%
# RMSE if 20.8

deep_feat_train <- h2o.deepfeatures(dl_model,train_hex[x_cols],layer = 2)
deep_feat_test <- h2o.deepfeatures(dl_model,test_hex[x_cols],layer = 2)
x_cols2 <- c(x_cols,colnames(deep_feat_train))
train_x2 <- cbind(train_x,as.data.frame(deep_feat_train))
test_x2 <- cbind(test_x,as.data.frame(deep_feat_test))

train_pool2 <- catboost.load_pool(data = train_x2[x_cols2],label = train_x2$Mean.Rank)
test_pool2 <- catboost.load_pool(data = test_x2[x_cols2])
cat_model2 <- catboost.train(train_pool2)
predicted_cat2 <- catboost.predict(cat_model2,test_pool2)

stats::cor(predicted_cat2,test_x$Mean.Rank)
## We get 80% correlation
regr.eval(predicted_cat2,test_x$Mean.Rank)
# The MAPE metric is 34%
# RMSE is 21.2

