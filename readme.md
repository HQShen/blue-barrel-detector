This is a color segmentation and barrel detection project.

We train a probabilistic color model from image data and use it to segment unseen images, detect a
blue barrel, and draw a bounding box around it. Two algorithms are used for this project. One is sigle Gaussian and the other is logistic regression.

Given the set of training images, you can hand-label examples of different colors by yourself using hand_label.py to circle the region of interest as training data. You can also use train_b.npy and train_nb.npy as training data.

Then you can use the training files for different ways of training and see how it performs in the test images afterwards. 

This is one of the results.
![image text](blue-barrel-detector/pics/1.png)
![image text](blue-barrel-detector/pics/2.png)
