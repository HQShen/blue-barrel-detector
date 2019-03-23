import numpy as np

alpha = 0.0001
train_b = np.load('train_b.npy')
train_nb = np.load('train_nb.npy')
# delete the first row of training data
train_b = np.delete(train_b,0,0)
train_nb = np.delete(train_nb,0,0)

trainset = np.concatenate((train_b/255,train_nb/255),axis = 0)

# set labels
y_b = np.ones([train_b.shape[0],1])
y_nb = -np.ones([train_nb.shape[0],1])
y = np.concatenate((y_b,y_nb),axis = 0)

w = np.zeros([3,2000])

# Gradient descent
for i in range(1,2000):
    temm = trainset.dot(w[:,i-1]).reshape(-1,1)
    logis = 1/(1+ np.exp(-y * temm))
    tem = trainset * y * (1- logis)
    w[:,i] = w[:,i-1] + alpha * np.sum(tem.T,axis = 1)

print(w[:,1999])


