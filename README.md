# Titanic-Model-Training

This is a project to experiment with some machine learning models to predict wheter a passenger from a Titanic dataset survived or not the accident.

By the end of the experiment, I was able to come up with both a Support Vector Machine and a FC Neural Network with an F1-score of about 99% on the test dataset.

For this, the libraries used were Pandas, Numpy, Sklearn, TensorFlow and MatPlotLib.

## DATA CLEANING

Several methods for data cleaning were apllied, such as imputation, scaling, one-hot encoding, feature selection, outliers handling and undersampling.

Of course, the training set was split into training, validation and testing.

## Machine Learning Models

I tried a bunch of different models using the following metrics: F1-score, precision, accuracy, recall and the confusion matrix.

The models tested were: Random Forest, Logistic Regression, Suport Vector Machine, Stochastic Gradient Discent Classifier and Gaussian Process.

By this point, the Support Vector Machine had already a 98% F1-score in the test set and adjusting the hyperparameters gave me a 99% F1-score by the end.

## Deep Learning Networks

I also used Tensorflow to create some different networks using different activation functions, architecture and optimizers. For this problem, FC layers were enough to get a 
great result and the Binary Cross Entropy loss function was used. The output layer alwaus used the Sigmoid activation function since this is a binary classification problem.

Models using Batch Normalization and Dropout layers were also trained are tested with different sizes and numbers of layers.

By the end, the best model found, hovering about 99% F1-score in the test set, used the Stochastic Gradient descent optimizer, with five layers of decresing number of neurons and the
ReLU activation function for all of the them. The Dropout layer was set to 0.3 and Batch Normalization did not show a lot of effectiveness.

