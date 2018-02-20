import numpy as np
from PIL import Image
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
    eval_labels = np.asarray(mnist.test.labels)

    # index = int(argv[1])

    # features = np.reshape(eval_data[index], [1, 784])
    # image = np.reshape(features, [28, 28])

    image = Image.open(argv[1])
    img_array = np.float32(np.array(image) / 255)

    features = np.reshape(img_array, [1, 784])

    pre_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": features},
        num_epochs=1,
        shuffle=False)

    results = mnist_classifier.predict(input_fn=pre_input_fn)

    for result in results:
        print ("\n\n\nprediction: {0}".format(result["classes"]))

    image.show()
    
if __name__ == "__main__":
    tf.app.run()