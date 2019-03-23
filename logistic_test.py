import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

image = cv2.imread(r'C:\Users\user\Desktop\courses\ECE 276A\ECE276A_HW1\trainset\46.png')
img = image[:, :, [2, 1, 0]]
w = np.array([-30.93211331, -19.26471414,  32.03887975])

a = img.shape[0]
b = img.shape[1]
img_2 = img.reshape(-1,3)
tem = img_2 .dot( w )
res = tem.reshape(a,b)
res[res > 0] = 1
res[res < 0] = 0
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
    
        if 1 < (maxr-minr)/(maxc-minc) < 2.3:
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-r', linewidth=1.5)
            print(bx,by)
            boxes.append([minc,minr,maxc,maxr])

plt.show()
