# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:23:43 2024

@author: sp3660
"""



#learning about correlations and similarities
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from numpy.linalg import norm

#random integer upt to a vale
x = np.random.randint(100)
#random float betwen 0 and 1
x = np.random.rand(5)
#array of random inttegers
x=np.random.randint(100, size=(5))



n=100
common_noise=np.random.normal(0,50,n)
x = np.linspace(0,np.random.randint(100),n)+np.random.normal(0,20,n)
y= np.linspace(0,np.random.randint(100),n)+np.random.normal(0,20,n)
st.pearsonr(x,y)
f,ax=plt.subplots(2,sharex=True)
ax[0].plot(x)
ax[0].plot(y)


cosine=np.sum(x*y)/(np.sqrt(np.sum(x**2))*np.sqrt(np.sum(y**2)))



norm_x = norm(x,keepdims=1)
norm_y= norm(y,keepdims=1)


x_norm=x/norm_x
y_norm=y/norm_y

ax[1].plot(x_norm)
ax[1].plot(y_norm)

#%%
from scipy.stats import norm 
import statistics 
stimuli=16
trials=100
stim_vector=np.linspace(0,stimuli,stimuli )
cell1=np.zeros([stimuli,trials])
cell2=np.zeros([stimuli,trials])

cell1_mean_responsess=np.random.normal(0,50,n)


x_dist=60*norm.pdf(stim_vector, 4.7, 1)
y_dist=60*norm.pdf(stim_vector, 7.3, 1)
y_dist[6]=x_dist[6]

for i,mn in enumerate(x_dist ):
    cell1[i,:]=np.random.normal(mn,2,trials) 
for i,mn in enumerate(y_dist ):
    cell2[i,:]=np.random.normal(mn,2,trials)


test_stim=7
f,ax=plt.subplots()
r,p=st.pearsonr(cell1[test_stim,:],cell2[test_stim,:])
ax.scatter(cell1[test_stim,:],cell2[test_stim,:])
ax.text(2,22,f'r: {np.round(r,3)}')

f,ax=plt.subplots()
ax.plot(cell1.mean(axis=1),'xb',alpha=0.8)
ax.plot(cell2.mean(axis=1),'or',alpha=0.8)

for i in range(cell1.shape[0]):
    ax.plot(np.ones(cell1[i, :].shape) * i,cell1[i, :] , '.b',markersize=1, alpha=0.1, label=f'Row {i}')
for i in range(cell1.shape[0]):
    ax.plot(np.ones(cell2[i, :].shape) * i,cell2[i, :] , '.r',markersize=1, alpha=0.1, label=f'Row {i}')



test_stim=7
f,ax=plt.subplots()
r,p=st.pearsonr(cell1.mean(axis=1),cell2.mean(axis=1))
ax.scatter(cell1.mean(axis=1),cell2.mean(axis=1))
ax.text(2,22,f'r: {np.round(r,3)}')