## TF_GradientTape 
This directory contains the same implementation of forward propagation as the regular implementation but the backpropagation is done using **gradient tape**.

## Testing_groud 
Here is all the testing of different layers on different datasets.

## layers.py
contains all the layers classes: **RNN, Conv2D, Dense, Dropout, MultiDense**

And different activation functions: **linear, relu, tanh, sigmoid, softmax**

Also impelments multiple initializations: **glorot_uniform, he_normal**

## model.py
here is the model class that implements the weight initialization, model fiting, prediction. It also contain different loss functions:

**mse, binary/categorica cross-entropy, cosine loss.**

## optimizers.py
contains **Adam** and **AdaGrad** optimizers.
