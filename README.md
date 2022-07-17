# Neural Network: Predicting Song Popularity on Spotify
## Overview
I built a neural network in R using keras; an interface for tensorflow. The goal was to tune this model to outperform or get close to a simply tuned XGBoost model, which only used bagging to determine hyper parameters. Ultimately, the neural network got close but did not outperform the XGBoost model, and took significantly longer to tune and train.

## Results
Below are the Mean Absolute Errors (MAE) for both models:
| XGBoost  | Neural Network |
| ------------- | -------------- |
| 13.83 | 13.91  |
### Graphs
Here are the correlation plots between fitted and actual values for each model:

![Correlations](https://user-images.githubusercontent.com/52394699/179418929-5fd90ad8-8701-498e-9cee-c6044cf98832.png)




Both models have very little spread in their fitted values. Looking at the distribution of the dependant variable below, song popularity, we can see that that both models cluster their predictions around the mode of about 60. This suggests to me that perhaps the dataset does not provide enough variability and that the best answer is simply picking the most common answer.

![Popularity distribution](https://user-images.githubusercontent.com/52394699/179418860-36b86191-6596-49dc-94ba-bc2edcd20c18.png)

### Models
The final NN model consisted of one dense layer and one dropout layer. The dropout layer was critical to prevent over-fitting. 2 dense and 2 dropout layers performed fairly well, but slightly overfit so the simpler model is utilised. 

The XGBoost model used 330 trees at an interaction depth of 14, which was selected by bagging. Even though cross validation is likely to give better predictive power, the in sample bagging method selected parameters that outperformed the NN.

## Conclusion
The neural network did not manage to outperform the much simpler and easier to tune XGBoost model. With more sophisticated tuning, the XGboost model would only get better than it already is. This is somewhat to be expected, as NN have been shown to be outperformed by XGBoost in tabular data. For instance, many Kaggle competitions are won with highly tuned XGBoost models, and not neural networks as they tend to overfit. 

However, this is somewhat of a worst case scenario for neural networks, with small tabular data. Given these limitations, with enough hyperparameter tuning it was able to get close to an XGBoost model. However once again, more sophisticated tuning of the XGBoost would almost certainly lead to a lower MAE.
