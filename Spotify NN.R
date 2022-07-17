keras::install_keras(tensorflow ='gpu')
library(keras)
tensorflow::tf_gpu_configured()
library(mlbench) 
library(dplyr)
library(magrittr)
library(neuralnet)
library(gbm)
library(tfruns)
# HYPER TUNING:
# https://www.youtube.com/watch?v=FscOZT0_ObA



setwd("C:/Users/Bret/OneDrive/R_Projects/TensorFlow/Spotify/")
data <- read.csv("spotify_data.csv")

data <- data[, -c(1,2,18)]
data <- data[,c(4, 1:3, 5:15)]
data$duration_ms <- data$duration_ms/60000
data$explicit <- ifelse(data$explicit == "True", 1, 0)

data %<>% mutate_all(as.numeric)

data <- as.matrix(data)
dimnames(data) <- NULL

data[,2:15] <- normalize(data[,2:15])
# Partition:
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7,0.3))
training <- data[ind==1, 2:15]
test <- data[ind==2, 2:15]
trainingtarget <- data[ind==1, 1]
testtarget <- data[ind==2, 1]

runs <- tuning_run("experiment.R",
                   echo = FALSE,
                   sample = 0.5,
                   flags = list(dense_units1 = c(20, 40, 80, 120, 160, 180),
                                dropout1 = c(0.1, 0.2, 0.3),
                                batch_size = c(16, 32)))

# best hyperparameter values:
head(runs)
results <- runs[,c(3,5:11)]
results <- results[order(results$metric_val_mae, results$metric_mae),]
head(results)


tensorflow::set_random_seed(1234)
model <- keras_model_sequential()
model %>%
  layer_dense(units = 160, activation = 'relu', input_shape = c(14)) %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1)

model %>% compile(loss = 'mse',
                  optimizer = 'rmsprop',
                  metrics = 'mae')

# Fit Model

mymodel <- model %>%
  fit(training,
      trainingtarget,
      epochs = 42,
      batch_size = 32,
      validation_split = 0.2,
      verbose=0)

# Evaluate
model %>% evaluate(test, testtarget)
pred <-  predict(model, test)
plot(testtarget, pred, pch = 19, cex=0.7, col='black', xlab='Actual Values', ylab= 'Fitted Values', main = "NN: Correlation between Fitted and Predicted")
abline(0,1, lwd=2, col='red')


n.var <- 14
n.tree.max <- 10000
oob.boost <- rep(NA, n.var)
trees.best <- rep(NA, n.var)
train <- as.data.frame(cbind(trainingtarget, training))
for(i in 1:n.var){
  mod <- gbm(trainingtarget ~ ., data = train, n.trees = n.tree.max, interaction.depth = i,
             shrinkage= 0.001, verbose=FALSE, distribution = 'gaussian')
  trees.best[i] <- gbm.perf(mod, plot.it=FALSE, oobag.curve=FALSE, method='OOB')
  oob.boost[i] <- mod$train.error[trees.best[i]]
}

d.best <- which.min(oob.boost)
n.tree.best <- trees.best[d.best]

boost.best <- gbm(trainingtarget ~ ., data = train, n.trees = n.tree.best, 
                  interaction.depth = d.best,
                  shrinkage= 0.001, verbose=FALSE,
                  distribution='gaussian')

testing <- as.data.frame(cbind(testtarget, test))
pred.boost <- predict(boost.best, newdata=testing)
plot(testtarget, pred.boost, pch = 19, cex=0.7, col='black',xlab='Actual Values', ylab= 'Fitted Values', main = "XGBoost: Correlation between Fitted and Predicted")
abline(0,1, lwd=2, col='red')

mean(abs(testing$testtarget - pred.boost))
