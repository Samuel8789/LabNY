# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:08:18 2022

@author: sp3660
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm

neurons=2000
n_patterns=200
patterns=np.random.randint(2, size=(neurons,n_patterns))

transp=patterns.T
weight=(2*patterns-1)@(2*transp-1)

weight=weight+weight[0,0]*np.diag(np.full(neurons,-1))  
np.std(weight)

frame=200
frames=np.zeros((neurons,frame))
frames[:,0]=np.random.randint(2, size=(neurons))
for i in range(1,frame):
    mult=weight@frames[:,i-1]
    frames[:,i]=np.where(mult>0,1, 0)
    
fig, ax=plt.subplots(1,3) 
ax[0].imshow(patterns, aspect='auto')
ax[1].imshow(weight, aspect='auto')
ax[2].imshow(frames[:,:int(frame/10)], aspect='auto')


test=np.diff(frames)
test2=np.sum(np.abs(test), axis=0)
plt.figure()
plt.plot(test2)
plt.ylim(0,50)

lastact=frames[:,-1]
similr = dot(lastact, patterns)/(norm(lastact)*norm(patterns))
plt.figure()
plt.plot(similr)


patterns[:,np.argmax(similr)]
lastact
mostsim=np.array([patterns[:,np.argmax(similr)],lastact])
plt.imshow(np.array([patterns[:,np.argmax(similr)],lastact]).T,aspect='auto')

