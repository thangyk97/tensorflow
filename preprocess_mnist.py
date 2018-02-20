from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

image = Image.open('/home/thangkt/git/mnist_png/testing/3/30.png')
img_array = np.array(image) / 255
print (img_array)
plt.imshow(img_array)
plt.show()


