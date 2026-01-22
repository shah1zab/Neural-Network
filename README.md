# Neural Network Project

## Overview

This project explores how a neural network behaves when applied to a binary classification problem based on geometric structure. The main goal is to understand how preprocessing, network depth, and model architecture influence learning, decision boundaries, and overall performance. Rather than focusing only on accuracy, the project emphasizes interpreting how the model transforms data internally and where it struggles.

## Dataset Description

The dataset consists of labeled training points belonging to one of two classes. Each point is classified based on whether it lies inside or outside a geometric shape. Because the boundary between classes is non linear, the dataset serves as a useful test case for examining the representational power of neural networks.

Initial visualizations of the data show overlapping regions and curved structures, suggesting that simple linear models would not perform well on this task.

## Data Preprocessing

Preprocessing is an essential step to ensure stable and meaningful training. Features are scaled so that they lie within similar ranges, preventing any single feature from dominating the learning process. Common approaches such as Min Max scaling and Z score standardization are appropriate for this type of data.

Any categorical variables are converted into numerical form using encoding techniques such as one hot encoding or label encoding, depending on the nature of the variable. The dataset is split into training and testing subsets, with 80 percent of the data used for training and the remaining portion used for evaluation.

Normalization rescales features to lie between zero and one, while standardization shifts features to have mean zero and unit variance. The choice between these techniques depends on the assumptions and sensitivities of the learning algorithm. Overall, proper preprocessing improves training stability, model interpretability, and reduces the risk of poor generalization.

## Model Architecture

The model is implemented as a sequential neural network composed of three fully connected dense layers. The hidden layers use the hyperbolic tangent activation function, which allows the network to learn non linear relationships. The final output layer uses a sigmoid activation function to produce probabilities suitable for binary classification.

The network contains a total of 25 trainable parameters. While relatively small, the architecture is sufficient to demonstrate how increasing depth introduces additional non linear transformations of the data.

## Training Procedure

Training is performed using the Keras framework. The model is optimized using stochastic gradient descent, with gradients computed through standard backpropagation. Rather than using the full dataset to estimate gradients, each update is based on a single randomly selected data point.

The loss function used is binary cross entropy, which corresponds to the negative log likelihood of a Bernoulli random variable. This choice is well suited for binary classification problems. The model is trained for 30 epochs with a batch size of one.

## Model Evaluation

After training, the final model predicts all data points as belonging to the same class, specifically classifying every point as outside the target shape. This indicates that the network failed to learn a useful decision boundary for the task.

Looking at intermediate representations reveals noticeable movement of data points between early hidden layers. However, increasing the number of layers does not lead to clearer separation between classes. Even with many hidden layers, the model does not converge to a meaningful classification boundary.

## Decision Boundary Analysis

Visualizations of the learned decision boundaries provide insight into how model complexity affects behavior. A model with no hidden layers produces a purely linear boundary. Adding one hidden layer allows the network to form a single curved region. With two hidden layers, multiple curved regions become possible.

Despite this added flexibility, the boundaries remain poorly aligned with the true structure of the data. Models with a large number of hidden layers often collapse into trivial solutions that classify all points into the same category.

## Overfitting Considerations

The model does not exhibit traditional overfitting, since performance remains weak even on the training data. However, examining intermediate models suggests a different form of overfitting, where the network learns artificial structures that do not reflect the true geometric boundary.

When trained on other datasets, the model shows similar behavior, producing decision boundaries that differ significantly from the true ones and generalizing poorly to unseen data.

## Conclusions

Each hidden layer introduces additional non linear transformations that, in theory, allow the model to capture more complex patterns. In practice, increased depth does not guarantee better performance, especially when the structure of the data does not align well with the model architecture.

This project highlights the importance of understanding how neural networks learn internally, not just how well they perform numerically. Examining intermediate layers and decision boundaries provides valuable insight into model limitations and helps explain why certain problems remain challenging for standard feedforward neural networks.
