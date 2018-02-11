from scipy.io import loadmat
import matplotlib.pylab as plt
import numpy as np 

ecg = np.loadtxt(
    '/home/thangkt/Downloads/PROCESSED ECG database/processedHighP/processedHighP_3_2.txt'
)

plt.plot(range(len(ecg)), ecg)
plt.show()











# Load ecg signal and index of R peaks
# data_ecg = loadmat('/home/thangkt/git/tensorflow/conv_ecg/100m.mat')
# data_index_R = loadmat('/home/thangkt/git/tensorflow/conv_ecg/i_100.mat')


# val = data_ecg['val']
# index_R = data_index_R['qrs_i_raw']

# for i in range(5):
#     temp = val[1, index_R[0][i+1] - 180: index_R[0][i+1] + 180]

#     fig = plt.figure(frameon=False)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)

#     plt.plot(range(temp.shape[0]), temp, c='grey')

#     fig.savefig('/home/thangkt/git/img_ecg/ecg' + str(i) + '.jpg',
#                 dpi=32
#     )

#     plt.close()

# print ("Successful!")