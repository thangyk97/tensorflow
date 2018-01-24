import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse

def sigmoid_func(x):
    """Compute and return the sigmoid activation 
    value for a given input value"""
    return 1 / (1 + np.exp(-x))

def gradient_func(X, y, W, learning_rate, epochs):
    # Initialize a list to store the loss value for each epoch
    loss_history = []

    """Loop over the desired number of epochs"""                        
    for epoch in np.arange(0, epochs):
        # take the dot product between our features 'X'
        # and the weight matrix 'X', then pass this value
        # through the sigmoid activation function, there by
        # giving us our predictions on the dataset
        preds = sigmoid_func(X.dot(W))

        # now that we have our predictions, we need to determine
        # our 'error', which is the difference between our
        # our predictions and the true values
        error = preds - y 

        # given our 'error', we can compute the total loss
        # value as the sum of squared loss -- ideally, our
        # loss should decrease as we continue training
        loss = np.sum(error ** 2)
        loss_history.append(loss)
        print ("[INFO] epoch #{}, loss:{:.7f}".format(epoch+1, loss))
        
        # The gradient update is therefore the dot product
        # between the transpose of 'X' and our error, scaled
        # by the total number of data points in 'X'
        gradient = X.T.dot(error) / X.shape[0]
        W += -learning_rate * gradient
    return W, loss_history

# Construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
                help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
args = vars(ap.parse_args())

# Generate a 2-class classification problem with 250 data
# points, where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=250, n_features=2, centers=2,
                    cluster_std=1.05, random_state=20)

#Insert a column of 1's as the first entry in the feature
# vector -- this is a little trick that allows us to treat
# the bias as a trainable parameter within the weight matrix
# rather than an entirely separate variable
X = np.c_[np.ones((X.shape[0])), X]

# Initialize our weight matrix such it has the same number of
# column as our input features
print ("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))

W, loss_history = gradient_func(X, y, W, args["alpha"], args["epochs"])

"""To demonstrate how to use our weght matrix as a classifier,
let's look over our a sample of training examples"""
for i in np.random.choice(250, 10):
    # Compute the prediction by taking the dot product of
    # the current feature vector with the weight matrix W,
    # then passing ti through the sigmoid activation func
    activation = sigmoid_func(X[i].dot(W))

    # the sigmoid func is defined over the range y=[0,1]
    # so we can use 0.5 as our threshold -- if 'activation' is
    # below 0.5, it's class '0'; otherwise it's class '1'
    label = 0 if activation < 0.5 else 1

    # show our output classification
    print ("acivation={:.4f}; predicted_label={}, true_label={}".format(
        activation, label, y[i]
    ))

# Compute the line of best fit by setting the sigmoid func
# to 0 and solving for X2 in terms of X1
Y = (-W[0] - (W[1]*X)) / W[2]

# plot the original data along with our line of best fit
plt.figure()
plt.scatter(X[:,1], X[:,2], marker="o", c=y)
plt.plot(X, Y, "r-")

# Construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), loss_history)
fig.suptitle("Training loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

