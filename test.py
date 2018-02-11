from scipy.io import loadmat
from scipy.ndimage import imread
import matplotlib.pylab as plt
import matplotlib.image as pmimg
import numpy as np 

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = imread('/home/thangkt/git/img_ecg/aaaa.jpg')

# img = rgb2gray(img)

fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

plt.imshow(img, cmap = 'gray')

# plt.savefig('/home/thangkt/git/img_ecg/aaaa.jpg',
#             dpi=32
# )

print (img.shape)