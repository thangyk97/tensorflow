import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 
from mnist_tutorial import cnn_model_fn




def main(argv):
        # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/home/thangkt/git/tensorflow/mnist_convnet_model")

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    index = int(argv[1])
    print (index)
    features = np.reshape(eval_data[index], [1, 784])
    image = np.reshape(features, [28, 28])

    pre_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": features},
        num_epochs=1,
        shuffle=False)

    results = mnist_classifier.predict(input_fn=pre_input_fn)

    for result in results:
        print ("prediction: {0}".format(result["classes"]))

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    tf.app.run()