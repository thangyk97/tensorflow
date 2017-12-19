import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


"""
Parameters: no_units # the number of unit in hidden layer
==========
"""
def nn_example(no_units, mnist):

    # Python optimisation variables
    learning_rate = 0.5
    epochs = 10
    batch_size = 100

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, 784])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random_normal([784, no_units], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([no_units]), name='b1')

    # W12 = tf.Variable(tf.random_normal([no_units, no_units], stddev=0.03), name='W12')
    # b12 = tf.Variable(tf.random_normal([no_units]), name='b12')


    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random_normal([no_units, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # now calculate the hidden layer output - in this case, let's use a softmax activated

    # hidden layer
    # hidden_out_2 = tf.add(tf.matmul(hidden_out, W12), b12)
    # hidden_out_2 = tf.nn.relu(hidden_out_2)


    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))


    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = - tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                     + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/home/thangkt/Desktop/AI/python/tensorflow')

    J = np.array([])
    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        print len(mnist.train.labels)
        total_batch = int(len(mnist.train.labels) / batch_size)


        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch

            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

            print mnist.test.images.shape
            print mnist.test.labels.shape

            summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            writer.add_summary(summary, epoch)
            J = np.append(J, avg_cost) # plot to check

        print("\nTraining complete!")
        writer.add_graph(sess.graph)
        result = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        sess.close()
        return result

       

if __name__ == "__main__":
    # run_simple_graph()
    # run_simple_graph_multiple()
    # simple_with_tensor_board()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    accuracy = nn_example(20, mnist)
    print accuracy


