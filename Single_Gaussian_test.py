import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

img = cv2.imread(r'C:\Users\user\Desktop\courses\ECE 276A\ECE276A_HW1\trainset\46.png')
img = img[:, :, [2, 1, 0]]
mu_b = np.array([ 22.40133256,  68.79046682, 128.99346339])
mu_nb = np.array([119.94242555, 103.68313491,  90.92928662])
sigma_b = np.array([[ 324.97949868,  174.76547283,   97.68073119],
       [ 174.76547283, 1217.34396821, 1654.1490144 ],
       [  97.68073119, 1654.1490144 , 2850.88132247]])
sigma_nb = np.array([[3787.19222926, 2709.94229967, 2702.44745783],
       [2709.94229967, 2904.06230099, 2855.45909252],
       [2702.44745783, 2855.45909252, 3123.38462824]])
p_b = 0.2688
p_nb = 0.7312

a = img.shape[0]
b = img.shape[1]

res = np.zeros([a*b,1])
img_2 = img.reshape(-1,3)
for i in range(img_2.shape[0]):
    p1 = 1/np.sqrt((2*np.pi)**3*(np.linalg.det(sigma_b)))*np.exp(-0.5*(img_2[i,:] - mu_b).dot(np.linalg.inv(sigma_b)).dot(img_2[i,:] - mu_b))*p_b
    p2 = 1/np.sqrt((2*np.pi)**3*(np.linalg.det(sigma_nb)))*np.exp(-0.5*(img_2[i,:] - mu_nb).dot(np.linalg.inv(sigma_nb)).dot(img_2[i,:] - mu_nb))*p_nb
    if p1 > p2:
        res[i] = 1

res = res.reshape([a,b])
# segmentation result
plt.imshow(res)
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))
opened = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

fig, ax = plt.subplots()
label_img = label(opened)
regions = regionprops(label_img)
ax.imshow(img, cmap=plt.cm.gray)
# set bounding boxes
boxes = []
for props in regions:
    if props.area >100:
        minr, minc, maxr, maxc = props.bbox
    
        if 1.3 < (maxr-minr)/(maxc-minc) < 2.3:
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-r', linewidth=1.5)
            print(bx,by)
            boxes.append([minc,minr,maxc,maxr])

plt.show()
