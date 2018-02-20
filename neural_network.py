import numpy as np
from scipy.io import loadmat
# from scipy.misc import imread
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

data = loadmat('ex3data1.mat')
X = np.array(data['X'])
y = np.array(data['y'])

img = X[4444,:]
img = np.reshape(img,(20,20), order='F')
plt.imshow(img)
plt.show()



####### Main 

# Number of neurons in each layer
input_num_units = 20*20
hidden_num_units = 500
output_num_units = 10

# Define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# Set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01

### Define weights and biases of the neural network
weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed = seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))

}

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)
output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

# Create session and run
with tf.Session() as sess:
    # Create initialized variables
    sess.run(init)

    # foreach epoch, do:
    #     foreach batch, do:
    #         create pre-processed batch
    #         run optimizer by feeding batch
    #         find cost and reiterate to minimize
    
    for epoch in range(epochs):

        avg_cost = 0
        total_batch = int(train.shape[0]/batch_size)

        for i in range(total_batch):
            batch_x, batch_y = batch_creator(
                batch_size,
                train_x.shape[0],
                'train'
            )
            _, c = sess.run(
                [optimizer, cost],
                feed_dict={x: batch_x, y: batch_y}
            )

            avg_cost += c / total_batch
        print ("Epoch:", (epoch+1), "cost =", "{: .5f}".format(avg_cost))

    print ("\nTraining complete!")

    # Find predictions on val set

