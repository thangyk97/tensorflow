

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)



def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")

  eval_data = mnist.test.images  # Returns np.array



  """softmax_tensor"""
  sess = tf.Session()

  

  saver = tf.train.import_meta_graph(
      '/home/thangkt/git/tensorflow/mnist_convnet_model/model.ckpt-20001.meta'
  )
  saver.restore(sess,tf.train.latest_checkpoint('/home/thangkt/git/tensorflow/mnist_convnet_model'))

  graph = tf.get_default_graph()


  op_to_restore = graph.get_tensor_by_name("softmax_tensor:0")
  
  print (sess.run(op_to_restore))



if __name__ == "__main__":
  tf.app.run()