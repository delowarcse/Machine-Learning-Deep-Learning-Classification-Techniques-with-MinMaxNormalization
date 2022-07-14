# Machine-Learning-Deep-Learning-Classification-Implementation with MinMax Normalization
This program is the implementation of Machine Learning and Deep Learning Techniques such as: Logistic Regression, Decision Tree, Random Forest, SVM, and DNN methods with MinMax Normalization ([0,1]-Normalizaton).

# Recommendation of use 
Please use Jupyter Notebook version of the code to better understand the all implemented mechine learning and deep learning techniques

# Logistic Regression (LR) 
I used a Logistic Regression model to classify each participant as a stroke or control based on their performance in a robotic task. For that purpose, I implemented a logistic regression classifier that was fitted in the binary logistic regression regularization. This regularization added a penalty as model complexity increased to ensure the model generalized the data and prevented overfitting with an increase in parameters. LR model assumes a linear relationship between the input features and output. The binary logistic model had a dependent variable with two possible outcomes as healthy control and stroke. 

# Decision Tree (DT)
I implemented a Decision Tree classifier as one of predictive modeling. It uses a tree-like model in which each internal node (non-leaf) is labeled with an input feature. The arcs coming from a node (branch) labeled with an input feature are labeled with each of the possible values of the target feature or the arcs leads to a subordinate decision node on a different input feature. Each leaf node is labeled with a class either healthy control or stroke. This model splits the nodes of all available features/parameters and then selects the splits, which results in the most homogeneous sub-nodes.

# Random Forest (RF) 
I implemented an ensemble learning model (i.e., a Random Forest classifier). It is a classification algorithm consisting of many decision trees, which uses bagging and feature randomness when building each individual tree. It tries to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree. The output of the random forest model was the class selected by most trees.

# Random Forest with Hyperparameters Tuning (RFT)
I tuned the hyperparameters (a hyperparameter is a parameter whose value is used to control the learning process) of the Random Forest model to determine the best hyperparameters. It relies more on experimental results than theory, and thus the best model to determine the optimal settings was by trying many different combinations to evaluate each model’s performance.

The tuned hyperparameters of the random forest model were: the number of trees in the forest, the maximum number of levels in each decision tree, the maximum number of features considered for spotting a node, the minimum number of data points placed in a node before the node is split, and the minimum number of data points allowed in a leaf node.

# Support Vector Machine (SVM) 
I implemented a Support Vector Machine (SVM) classifier. It constructed a set of hyperplanes (hyperplanes are decision boundaries that help to classify the data points) in high-dimensional space to perform the classification task. The model transformed the data to find an optimal boundary between outputs (control or stroke). A good separation is achieved by the hyperplane that had the largest distance, or functional margin, to the nearest training data point of any class.

# Deep Neural Network (DNN)
I also implemented a Deep Learning technique, namely, Deep Neural Network (DNN). It is a part of a broader family of machine learning techniques based on artificial neural networks.

My DNN classifier implementation consisted of three hidden layers between input and output layers. The first hidden layer had 12 units with the Rectified Linear Unit (ReLU) as the activation function, the second hidden layer had 8 units with the ReLU as the activation function, and the third hidden layer had 1 unit with the sigmoid function as the activation function. We also used: binary cross-entropy as loss function, the Root Means Square propagation optimizer (RMSprop), the batch size of 10, and the number epoch of 100. An epoch refers to the number of passes of the entire training dataset the deep learning technique has completed. The input layer had 12 units for 12 features, and the output had 1 unit to predict a 0 or 1 that maps back to the “healthy control” or “stroke” class. Each layer of nodes trained a distinct set of features based on the output of the previous layer. The feature hierarchical process of our DNN model made it capable of handling very large, and high-dimensional datasets with billions of parameters passed through nonlinear functions.
