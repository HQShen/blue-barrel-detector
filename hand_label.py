import numpy as np
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import cv2

n = 42 #first 42 as input training set
train_nb = np.empty([1,3])
for i in range(1,n+1):
    image = cv2.imread(r'C:\Users\user\Desktop\courses\ECE 276A\ECE276A_HW1\trainset'+ '\%d'%(i)+'.png')
    img = image[:, :, [2, 1, 0]]
    plt.imshow(img)
    my_roi = RoiPoly(color='r')
    plt.show()
    img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = my_roi.get_mask(img_GRAY)
    # train_b and train_nb are used separately as positive and negative samples.
    # train_nb = np.concatenate((train_nb, img[mask]), axis=0)
    train_b = np.concatenate((train_b, img[mask]), axis=0)
    
