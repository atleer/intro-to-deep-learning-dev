# Plan for a Three-Day Intro to Deep Learning Workshop

## Day 1: Create and train neural network (without OOP)

### Unit 1: PyTorch Basics
- What are Tensors? - Create tensors
- Initialize tensors - `torch.linspace`
- Go from tensors to other datatypes (numpy arrays, lists)
- Arithmetic operations on tensors

### Unit 2: Create a Neural Network (without Object-Oriented Programming)
- Creating layers of neurons + activation functions - `nn.Linear(in_features, out_features)`
- Forward pass of input through layers. Loss function. - `net.forward(input_data)`, `nn.CrossEntropyLoss()`
- Create optimizer - `torch.optim.SGD(model.parameters())`
- Calculate accuracy - `(pred_labels == true_labels).sum()/len(true_labels)`

### Unit 3: Train a Neural Network Part 2. Functions.
- General function section - `def`
- Put code to train network in functions
- Put code to train network in functions - but add `argument = value` to function calls and `argument = default_value` to function def 

### Unit 4: Train a Neural Network Part 1. For-Loops
- Update network parameters (i.e. train) without loop - `loss.backward()`, `optimizer.step()`
- General for loop section
- Train with iterations: Put update of network parameters inside for loop through epochs



## Day 2: Create and train neural network with OOP. Prepare data for training

### Unit 5: Intro to Object-Oriented Programming
- Create Class and Object
- Inheritance - `super().__init__()`

### Unit 6: Create a Neural Network (with Object-Oriented Programming)
- Make model a class
- Apply to other data than day 1 - wine dataset. 
- Regression in addition to classification?

### Unit 7: Prepare Data for Training
- The `__init__` function - what goes into it? Data loading and creating tensors
- The `__len__` and `__getitem__` functions - together with `__init__` the three essential parts of the custom data class
- The `torch.utils.data.DataLoader` - batching and shuffling data for training
- Split into train, validation, and test data - `torch.utils.data.random_split()`

### Unit 8: Improve Network - Reduce Bias, Tune Hyperparameters, Data Augmentation
- Scale Data - `StandardScaler()`
- Train with dataloader - `for (features_batch, labels_batch) in train_dataloader`
- Data augmentation - add gaussian noise, randomly drop features
- Batch normalization

## Day 3: Different kinds of neural network architectures. Apply to neural data.

### Unit 9: Dense (feed forward)
- Tune hyperparameters (learning rate etc.)
- Deep vs. wide network
- Regularization?

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