# Plan for a Three-Day Intro to Deep Learning Workshop

## Day 1: Create and train neural network (without OOP)

### Unit 1: PyTorch Basics
- Create tensors
- Go from tensors to other datatypes (numpy arrays, lists)
- Arithmetic operations on tensors

### Unit 2: Create a Neural Network (without Object-Oriented Programming)
- Creating layers of neurons. Add activation fun
- Forward pass of input through layers. Loss function
- Create optimizer
- Calculate accuracy

### Unit 3: Train a Neural Network Part 1. For-Loops
- Update network parameters (i.e. train) without loop
- General for loop section
- Train with iterations: Put update of network parameters in for loop through epochs

### Unit 4: Train a Neural Network Part 2. Functions.
- General function section
- Put code to train network in functions

## Day 2: Create and train neural network with OOP. Prepare data for training

### Unit 5: Intro to Object-Oriented Programming
- Polymorphism
- Inheritance

### Unit 6: Create a Neural Network (with Object-Oriented Programming)
- Make model a class
- Apply to other data than day 1 - wine dataset. 
- Regression in addition to classification?

### Unit 7: Prepare Data for Training
- Download standard datasets from Torchvision and make dataloaders (single function)
- Transform data (to tensors, normalization)
- Prepare your own data (requires OOP)

### Unit 8: Improve Network - Reduce Bias, Tune Hyperparameters, Data Augmentation
- Split into train, validation, and test data
- Tune hyperparameters (learning rate etc.)
- Data augmentation
- Regularization

## Day 3: Different kinds of neural network architectures. Apply to neural data.

### Unit 9: Dense (feed forward)
- Deep vs. wide network
- Regularization 2?
- Batch normalization

### Unit 10: RNN
- For time series - forecasting
- Apply to Ephys Data?

### Unit 11: CNN
- Great for image data (but has ton of applications)
- Apply to Ca-imaging dataset from neuromatch
- Use GPU - google colab


# Day 1 (programming technique: loops and functions)
  - `torch.tensor()`
  -  Create a Neural Network (without OOP), Calculate Accuracy
  -  `def` - wrap things in functions before loops 
    - can then use utility functions
  -  `loop`, `optimizer`, `loss` - Train a Neural Network


# Day 2 (programming technique: OOP)
  - `class` (inheritance & polymorphism)
  - `Model`
  - `DataLoader`, `Dataset`
  - Improving Your Neural Network: Reducing Bias, Augmenting, etc (Split into train and test data)

# Day 3 Designing Neural Networks for Neuro
  - Dense
  - Recurrent
  - Convolutional
  - (Cool application: What can we do with this; RSA representational similarity analysis as special topic)