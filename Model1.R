setwd('/home/yasar/sling_media/')

train <- read.csv('data/transformed_data.csv')
library(DMwR)

train_rows <- sample(1:101,70)
train_x <- train[train_rows,3:3174]
test_x <- train[-train_rows,3:3174]
x_cols <- colnames(train_x)[1:3172]
library(corrplot)
correlations <-cor(train_x)
correlations_df <- as.data.frame(correlations)

library(reshape2)
req <- subset(train,select = x_cols, Mean.Rank>=10 & Mean.Rank<20)
req1<- melt(req,id.vars = 'Country.Name')
ggplot(data=req1, aes(value)) + 
  geom_histogram(aes(y =..density..), 
                 breaks=seq(0, 110, by = 2), 
                 col="Blue", 
                 fill="green", 
                 alpha = .2) + 
  geom_density(col=2) + 
  labs(title="Rank for Countries") +
  labs(x="Rank", y="Count") + facet_grid(~Country.Name)

above_60 <- subset(correlations_df, abs(Mean.Rank)>=0.6)
above_60 <- subset(above_60, select = rownames(above_60))
correlations_df <- subset(correlations_df, abs(Mean.Rank)<0.6 | Mean.Rank==1)
correlations_df <- subset(correlations_df, select = c(rownames(correlations_df),'Mean.Rank'))
corrplot.mixed(as.matrix(above_60))

above_55 <- subset(correlations_df, abs(Mean.Rank)>=0.55)
above_55 <- subset(above_55, select = rownames(above_55))
correlations_df <- subset(correlations_df, abs(Mean.Rank)<0.55 | Mean.Rank==1)
correlations_df <- subset(correlations_df, select = c(rownames(correlations_df),'Mean.Rank'))
corrplot.mixed(as.matrix(above_55))


above_50 <- subset(correlations_df, abs(Mean.Rank)>=0.5)
above_50 <- subset(above_50, select = rownames(above_50))

library(randomForest)
rf_model <- randomForest(train_x[x_cols],train_x$Mean.Rank)
predicted_rf <- predict(rf_model, test_x)
stats::cor(predicted_rf,test_x$Mean.Rank)
## We get 80% correlation
regr.eval(predicted_rf,test_x$Mean.Rank)
# The MAPE metric is 34.8%
# RMSE if 21.2
feature_importance <- as.data.frame(rf_model$importance)
feature_importance$column_name <- rownames(feature_importance)
feature_importance$correlation_value <- sapply(feature_importance$column_name, FUN=function(x) correlations_df['Mean.Rank',x])
feature_importance <- feature_importance[with(feature_importance, order(-IncNodePurity,-correlation_value)),]

correlations_df <- as.data.frame(correlations)
top_features_correlations <- subset(correlations_df, select = feature_importance$column_name[1:20], rownames(correlations_df) %in% feature_importance$column_name[1:20])

corrplot.mixed(as.matrix(top_features_correlations))

library(ggplot2)
library(gganimate)

top_countries <- subset(train,select = c('Country.Name',feature_importance$column_name[1:20]), Mean.Rank < 10 )
second_top_countries <- subset(train,select = c('Country.Name',feature_importance$column_name[1:20]), Mean.Rank >= 10 & Mean.Rank < 20 )

top_countries <- top_countries[order(top_countries$Mean.Rank),]
second_top_countries <- second_top_countries[order(second_top_countries$Mean.Rank),]

top_countries <- melt(top_countries, id=c('Country.Name','Mean.Rank'))

ggplot(top_countries,aes(x=variable,y=value,fill=variable)) + 
  geom_bar(stat = 'identity')  + 
  ggtitle('Ranking of Top Countries') + facet_wrap(~Country.Name) +
  geom_line(stat='identity',aes(y=Mean.Rank,group=1), color = 'grey',size=0.5)

p <- ggplot(top_countries) + 
  geom_bar(stat = 'identity',position = 'identity',aes(x=Country.Name,y=value,frame=variable,cumulative=F,fill=variable))  + 
  ggtitle('Ranking of Top Countries') +
  labs(x='Country', y='Rank') +
  geom_line(aes(x=Country.Name,y=Mean.Rank,group=1),color='red',size=2.5, stat='identity')


library(animation)
ani.options(interval=0.5)
gganimate(p)
gganimate(p,filename = 'Plots/Top_Features.gif')

second_top_countries <- melt(second_top_countries, id=c('Country.Name','Mean.Rank'))

ggplot(second_top_countries,aes(x=variable,y=value,fill=variable)) + 
  geom_bar(stat = 'identity')  + 
  ggtitle('Ranking of Top Countries') + facet_wrap(~Country.Name) +
  geom_line(stat='identity',aes(y=Mean.Rank,group=1), color = 'grey',size=0.5)

p2 <- ggplot(second_top_countries) + 
  geom_bar(stat = 'identity',position = 'identity',aes(x=Country.Name,y=value,frame=variable,cumulative=F,fill=variable))  + 
  ggtitle('Ranking of Top Countries') +
  labs(x='Country', y='Rank') +
  geom_line(aes(x=Country.Name,y=Mean.Rank,group=1),color='red',size=2.5, stat='identity')

ani.options(interval=0.5)
gganimate(p2)
gganimate(p2,filename = 'Plots/Top_Features2.gif')


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
## We get 55% correlation
regr.eval(as.data.frame(predicted)$predict,test_x$Mean.Rank)
# The MAPE metric is 39%
# RMSE if 27.8


library(catboost)
train_pool <- catboost.load_pool(data = train_x[x_cols],label = train_x$Mean.Rank)
test_pool <- catboost.load_pool(data=test_x[x_cols])
cat_model <- catboost.train(train_pool)
predicted_cat <- catboost.predict(cat_model,test_pool)

stats::cor(predicted_cat,test_x$Mean.Rank)
## We get 84% correlation
regr.eval(predicted_cat,test_x$Mean.Rank)
# The MAPE metric is 28%
# RMSE if 18.2

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
## We get 81% correlation
regr.eval(predicted_cat2,test_x$Mean.Rank)
# The MAPE metric is 33%
# RMSE id 20.2


