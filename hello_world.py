import tensorflow as tf 

# Build computational graph
a = tf.placeholder(tf.int16)
b = a * 2


# Create session and run the graph
with tf.Session() as sess:
    
    print sess.run(b, feed_dict={a:[1,2,3]})

# Close session
sess.close()