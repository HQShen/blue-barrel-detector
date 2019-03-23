import numpy as np

train_b = np.load('train_b.npy')
train_nb = np.load('train_nb.npy')

train_b = np.delete(train_b,0,0)
train_nb = np.delete(train_nb,0,0)

mu_b = np.sum(train_b,axis=0)/train_b.shape[0]
train_b = train_b-mu_b
sigma_b = np.zeros([3,3])
for i in range(train_b.shape[0]):
    sigma_b = sigma_b + np.multiply(train_b[i,:].reshape(1,3),train_b[i,:].reshape(3,1))/train_b.shape[0]

mu_nb = np.sum(train_nb,axis=0)/train_nb.shape[0]
train_nb = train_nb-mu_nb
sigma_nb = np.zeros([3,3])
for i in range(train_nb.shape[0]):
    sigma_nb = sigma_nb + np.multiply(train_nb[i,:].reshape(1,3),train_nb[i,:].reshape(3,1))/train_nb.shape[0]

p1 = train_b.shape[0]/(train_b.shape[0]+train_nb.shape[0])
p2 = train_nb.shape[0]/(train_b.shape[0]+train_nb.shape[0])

print('mu_b :',mu_b,'\nmu_nb :',mu_nb,'\nsigma_b :',sigma_b,'\nsigma_nb :',sigma_nb,'\np_b :',p1,'\np_nb :',p2)
