import numpy as np 
import tensorflow as tf 

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # reshape X to 4D tensor
    # [batch_size, width, height, channels]
    # gray image ==> 1 channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,    # [batch_size, 28, 28, 1] -> [batch_size, 28, 28, 32]
        filters=32,            # Computes 32 features
        kernel_size=[5, 5],    # 5x5 filter
        padding="SAME",        # preserve width and height
        activation=tf.nn.relu,)# ReLU activation

    # [batch_size, 28, 28, 32] => [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2,)

    # [batch_size, 14, 14, 32] => [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding='SAME',
        activation=tf.nn.relu,)

    # [batch_size, 14, 14, 64] => [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2,)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu,)

    # Add dropout operation
    # 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=(mode == tf.estimator.ModeKeys.TRAIN),)

    logits = tf.layers.dense(
        inputs=dropout,
        units=10,)

    predictions = {
        "classes": tf.argmax(
            input=logits,
            axis=1,),
        "probabilities": tf.nn.softmax(
            logits,
            name="softmax_tensor",),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,)
    
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits,)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimize = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimize.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,)
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"]),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops,)

def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    id_rand = np.arange(train_data.shape[0])
    np.random.shuffle(id_rand)
    train_data = train_data[id_rand]
    train_data = train_data[0: 6500]

    train_labels = train_labels[id_rand]
    train_labels = train_labels[0: 6500]


    eval_data = mnist.test.images # np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="/tmp/mnist_convnet_model",)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with the 
    # label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50,)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True,)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=500,)
        # hooks=[logging_hook],)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False,)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print (eval_results)

if __name__ == "__main__":
    tf.app.run() 





