setwd('/home/yasar/sling_media/')

train <- read.csv('data/transformed_data.csv')
library(DMwR)
library(infotheo)
rice_rule_bins <- ceiling(202**0.33)

train$Mean.Rank <- as.factor(as.character(unlist(discretize(train$Mean.Rank,'equalwidth',rice_rule_bins))))
train_rows <- sample(1:101,70)
train_x <- train[train_rows,3:3174]
test_x <- train[-train_rows,3:3174]
x_cols <- colnames(train_x)[1:3172]

library(randomForest)

rf_model <- randomForest(x=train_x[x_cols],y = train_x$Mean.Rank)
predicted <- predict(rf_model, test_x)
cm <- table(test_x$Mean.Rank,predicted)
library(caret)
confusionMatrix(cm)
# Accuracy 22%

library(catboost)
train_pool <- catboost.load_pool(data = train_x[x_cols],label = as.numeric(train_x$Mean.Rank))
test_pool <- catboost.load_pool(data = test_x[x_cols])

cb_model <- catboost.train(train_pool)
predicted_cb <- catboost.predict(cb_model,test_pool)
predicted_cb <- round(predicted_cb)
cm_cb <- table(as.numeric(test_x$Mean.Rank),predicted_cb)
cm_cb <- cbind(cm_cb,rep(0,6))
dimnames(cm_cb)[[2]][6]<-'6'
confusionMatrix(cm_cb)
# 35% accuracy

