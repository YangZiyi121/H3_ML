from tkinter import W
from turtle import forward, shape
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle

# do not change the variable names
NAME = "Ziyi Yang" #replace with your name
ID = "181989" #replace with your KAUST ID

""" your code is graded automatically, only write code in where you are told to do """

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in<int>: number of nodes of the input layer
    # n_out<int>: number of nodes of the output layer

    # Output:
    # W<numpy array>: matrix of random initial weights with size (n_out x (n_in + 1))"""


    ''' you do not need to modify this function '''
    ''' notice that addtional 1 is added, this is for the bias parameters'''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W

def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    ''' you need to figure out the result '''
    result = 1 / (1 + np.exp(-z))
    return result

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data<numpy array>: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label<numpy array>: vector of label corresponding to each image in the training
       set
     validation_data<numpy array>: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label<numpy array>: vector of label corresponding to each image in the
       training set
     test_data<numpy array>: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label<numpy array>: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    #train, trainLabels, valid, validLabels, test, testLabels = ...

    # load the original data set to a Dictionary
    mat = loadmat("mnist_all.mat")

    # have a look at the dictionary's keys, you need to observe the shape to know what stands for a single data example
    print("all of the keys in mat: ")
    for key in mat.keys():
        if key.startswith("train") or key.startswith("test"):
          print( key, mat[key].shape, mat[key].dtype )
    

    ''' you need to implement the follow functionalities '''
    # 1. extract the training to big numpy array, testing data to big numpy array;
    #    forming the right labels for each data example, you can figure out which label is for which data group by oberserving the keys
    #    in the dictionary;
    #    some useful doc:
    #    https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html?highlight=concatenate#numpy.concatenate 
    train = mat["train0"]
    test = mat["test0"]
    trainLabels = np.zeros(mat["train0"].shape[0])
    testLabels = np.zeros(mat["test0"].shape[0])
    for key in mat.keys():
      if key.startswith("train") and (key != "train0"):
        train = np.concatenate((train, mat[key]), axis = 0)
        trainLabels = np.concatenate((trainLabels, [int(key[-1]) for _ in range(mat[key].shape[0])]), axis = None)
      if key.startswith("test") and (key != "test0"):  
        test = np.concatenate((test, mat[key]), axis = 0)
        testLabels = np.concatenate((testLabels, [int(key[-1]) for _ in range(mat[key].shape[0])]), axis = None)
    trainLabels = trainLabels.astype(int) 
    testLabels = testLabels.astype(int)   



    # 2. doing numerical operation need operand to be consistent, you need to convert the uint8 type to float64 type
    #    for each entry in the big numpy array, check the numpy doc: https://numpy.org/doc/stable/reference/arrays.scalars.html?highlight=numpy%20uint8#scalars
    #    https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html?highlight=array%20astype#numpy.ndarray.astype
    train = train.astype(float)
    test = test.astype(float)
    allData = np.concatenate((train, test), axis = 0)
    
    # 3. normalize the data to be in the range [0, 1] for each entry in the big numpy array
    train = train / 255
    test = test / 255

    # 4. we need to fetch some data from the training data as the validation data, one way is to randomly fetch some data example
    #    another way is to randomly shuffle the training data and then pick the first sevaral data example as the training data
    #    and the rest as the validation data, you can do both ways to your convenience
    #    note: when doing operation like changing order on the numpy data array, 
    #          you need to do the same operation on the numpy label array, because each data has a corresponding label
    idx = np.random.permutation(len(train))
    train_shuffled, trainLabels_shuffled = train[idx], trainLabels[idx]
    train = train_shuffled[:54000]
    trainLabels = trainLabels_shuffled[:54000]
    valid = train_shuffled[54000:]
    validLabels = trainLabels_shuffled[54000:]

    # 5. if you print the entire data examples, for some position in the data array, you notice that they are all 0,
    #    this feature corresponds to that position is of no help to gaining information for the classification problem and
    #    including those zeros in the numerical operation will include unnecessary computational cost, you need to remove those
    #    for all of the data examples. some useful docs:
    #    https://numpy.org/doc/stable/reference/generated/numpy.all.html?highlight=all#numpy.all
    #    https://numpy.org/doc/stable/reference/generated/numpy.delete.html?highlight=delete#numpy.delete
    columnToD = []
    for i in range(allData.shape[1]):
      if np.all(allData[:, i] == 0) == True:
        columnToD.append(i)
    train = np.delete(train, columnToD, 1)
    test = np.delete(test,columnToD, 1)
    valid = np.delete(valid,columnToD, 1)


    # 6. make sure that you convert the entries in the label arrays to be integer, otherwise, this will type error for
    #    later computation. convince youself that using floating number to represent label makes no sense

    # return the processed data
    return train, trainLabels, valid, validLabels, test, testLabels


def nnObjFunction(params, *args):
    """ nnObjFunction computes the value of objective function (negative log
       likelihood error function with regularization) given the parameters
       of Neural Networks, thetraining data, their corresponding training
       labels and lambda - regularization hyper-parameter.

     Input:
     params: vector of weights of 2 matrices w1 (weights of connections from
         input layer to hidden layer) and w2 (weights of connections from
         hidden layer to output layer) where all of the weights are contained
         in a single vector.
     n_input: number of node in input layer (not include the bias node)
     n_hidden: number of node in hidden layer (not include the bias node)
     n_class: number of node in output layer (number of classes in
         classification problem
     training_data: matrix of training data. Each row of this matrix
         represents the feature vector of a particular image
     training_label: the vector of truth label of training images. Each entry
         in the vector represents the truth label of its corresponding image.
     lambda: regularization hyper-parameter. This value is used for fixing the
         overfitting problem.

     Output:
     obj_val: a scalar value representing value of error function
     obj_grad: a SINGLE vector of gradient value of error function
     NOTE: how to compute obj_grad
     Use backpropagation algorithm to compute the gradient of error function
     for each weights in weight matrices.

     reshape 'params' vector into 2 matrices of weight w1 and w2
     w1: matrix of weights of connections from input layer to hidden layers.
         w1(i, j) represents the weight of connection from unit j in input
         layer to unit i in hidden layer.
     w2: matrix of weights of connections from hidden layer to output layers.
         w2(i, j) represents the weight of connection from unit j in hidden
         layer to unit i in output layer."""


    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    """ note that we only have w1 and w1, but the fact is that the bias parameters(b1,b2) are already included in w1 and w2
        you can observe the dimension of w1 and w2, addtiional 1 is added to n_input and n_hidden.
        so you need to manually add a column of 1 to the input data for each layer to do the matrix multiplication
        
        to do matrix multiplication, you will need to guarantee the dimension of the matrix is well-matched.
        this can guide you figure out which matrix is before which.
        from input layer to the final output, dimensions of matrix is something like:

        number_of_data_example * (number_of_node_in_input_layer + 1)
        -> number_of_data_example * (number_of_node_in_hidden_layer + 1)
        -> number_of_data_example * number_of_class

        if you combine the information on the dimension of w1 and w2, you can figure out how to write matrix multiplication
        """

    #obj_val, obj_grad = ...


    ''' you need to implement the follow functionalities '''

    # 1. feedforward pass from input layer to hidden layer
    #    note: remember to add the bias before doing matrix multiplication
    bias1 = np.ones(training_data.shape[0], dtype = float)
    training_data = np.c_[training_data, bias1]
    a = np.dot(training_data, w1.T)  #linear_1
    z = sigmoid(a) #activate_1

    # 2. feedforward pass from hidden layer to output
    #    note: remember to add the bias before doing matrix multiplication
    bias2 = np.ones(z.shape[0], dtype = float)
    z_b = np.c_[z, bias2]  
    b = np.dot(z_b, w2.T) #linear_2
    o = sigmoid(b) #activate_2 output

    # 3. compute the object function or the loss term
    training_label_one_hot = np.zeros((training_label.size, training_label.max()+1))
    training_label_one_hot[np.arange(training_label.size),training_label] = 1
    loss_per_unit = -np.sum((training_label_one_hot * np.log(o) + (1 - training_label_one_hot) * np.log(1-o)), axis=1)
    loss = (1/len(training_label)) * np.sum(loss_per_unit)
    # print("loss", loss)

    # 4. add the regularization term to the loss function to form a final loss term
    #    note: do not make mistake on the positities for each term
    obj_val = loss + (lambdaval/(2 * len(training_label)))*(np.sum(w1**2) + np.sum(w2**2))

    # 5. compute the gradient of the loss term with respect to all the weights
    #    note:  using backpropagation algorithm to compute the gradient of loss term, 
    # # Applying equations from assignment document       
    error = o - training_label_one_hot
    dw2 = (1/len(training_label))*error.T @ z_b
    error_w2 = error @ w2 [:,: n_hidden]
    dev_sigmoid = (1-z)*z
    dw1 = (1/len(training_label)) * (dev_sigmoid * error_w2).T @ training_data
    obj_grad = np.concatenate((dw1.flatten(), dw2.flatten()), 0)
    return obj_val, obj_grad

def nnPredict(w1,w2,data):

    """ nnPredict predicts the label of data given the parameter w1, w2 of Neural
     Network.

     Input:
     w1: matrix of weights of connections from input layer to hidden layers.
        w1(i, j) represents the weight of connection from unit i in input
         layer to unit j in hidden layer.
     w2: matrix of weights of connections from hidden layer to output layers.
         w2(i, j) represents the weight of connection from unit i in input
         layer to unit j in hidden layer.
     data: matrix of data. Each row of this matrix represents the feature
           vector of a particular image

     Output:
     label: a column vector of predicted labels"""

    ''' you need to implement the follow functionalities '''
    # just like the forward pass in the nnObjFunction, you need to forward pass the data to get its label
    # note: you need to figure out which labe it is base on the output of the final layer, should be integer corresponding to the label
    #       you specified in the preprocess() funciton

    bias1 = np.ones(data.shape[0], dtype = float)
    data = np.c_[data, bias1]
    a = np.dot(data, w1.T)
    z = sigmoid(a)

    bias2 = np.ones(z.shape[0], dtype = float)
    z_b = np.c_[z, bias2]
    b = np.dot(z_b, w2.T)
    o = sigmoid(b)
    predicted_label = np.argmax(o, axis=1).T
    return predicted_label

def main():
    
  """**************Neural Network Script Starts here********************************"""
  start = time.time()

  train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();
  print(train_data.shape)
  print(validation_data.shape)
  print(test_data.shape)



  ''' Train Neural Network'''

  ''' nerwork setup(no need to modify) '''
  # set the number of nodes in input unit (not including bias unit)
  n_input = train_data.shape[1]
  # set the number of nodes in hidden unit (not including bias unit)
  n_hidden = 20; # you can play with this setting
  # set the number of nodes in output unit
  n_class = 10;# this has to be 10, because we have 10 classes
  # initialize the weights into some random matrices
  initial_w1 = initializeWeights(n_input, n_hidden)
  initial_w2 = initializeWeights(n_hidden, n_class)
  # unroll 2 weight matrices into single column vector
  initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
  # set the regularization hyper-parameter
  lambdaval = 0.2;# you need to play with this setting to get better validation accuracy
  # form the parameter tuple
  args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

  # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
  opts = {'maxiter': 50} # Preferred value. you can play with this setting
  nn_params = minimize(nnObjFunction, initialWeights, jac = True, args = args, method = 'CG', options = opts)
  """ In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
      and nnObjGradient. Check documentation for this function before you proceed.
      nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)"""
  
  # Reshape nnParams from 1D vector into w1 and w2 matrices
  w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
  w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

  ''' see how the trained model perform on training data ''' 
  predicted_label = nnPredict(w1,w2,train_data)
  print("n_hidden: " + str(n_hidden) + " lambda: " + str(lambdaval) + "\n")
  print('\nTraining set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

  ''' see how the trained model perform on validation data '''
  predicted_label = nnPredict(w1,w2,validation_data)
  print('\nValidation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

  ''' see how the trained model perform on test data '''
  predicted_label = nnPredict(w1,w2,test_data)
  print('\nTest set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

  end = time.time()
  print(f"time consumed for one run: {round(end-start,3)} seconds")

  ''' save the trained parameters to a file on the disk to be later use '''
  ''' and we check the test accuray using the parameter you saved on this file '''
  params = {"n_hidden": n_hidden, "w1": w1, "w2": w2, "lambda": lambdaval}
  pickle.dump(params, open("params.pickle", "wb"))
  print("Params dumped")

if __name__ == "__main__":
    # do not modify below, you can modify your training inside the main() function when doing your owne hyper parameter tuning
    # if you directly run this file, it will run the main() function
    main()
