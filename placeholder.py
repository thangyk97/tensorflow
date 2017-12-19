import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

# First, load the image again
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/OrchidSlice.png"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("float", [None, None, 4])
gray_img = np.zeros(raw_image_data.shape[0:2])

def convert_gray(image):
    for i in range (gray_img.shape[0]):
        for j in range (gray_img.shape[1]):
            gray_img[i,j] = np.dot(image[i,j,0:3], np.array([0.21,0.72,0.07]))
    return gray_img

gray_img = convert_gray(raw_image_data)

# with tf.Session() as session:
#     result = session.run(slice, feed_dict={image: raw_image_data})
#     print(result.shape)


plt.imshow(gray_img)
plt.show()
