# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 18:15:56 2022

@author: sp3660
"""

from math import e, pi, atan
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,pi,10)
y=np.sin(x)+1j*np.cos(x)
plt.scatter(x,y)


num_pts=20 # number of points on the circle
ps=np.arange(num_pts+1)
psmine = np.linspace(0,2*pi,20)
psmine2 = np.linspace(0,1,20)

# j = np.sqrt(-1)
pts = np.exp(2j*np.pi* (ps)/num_pts)
ptsmine = np.exp(1j*psmine)
ptsmine2 = np.exp(2j*pi*(psmine2))

angle=atan(ptsmine.imag[1]/ptsmine.real[1])

fig, ax = plt.subplots(1)
ax.plot(pts.real[:-1], pts.imag[:-1] , '-o')
ax.set_aspect(1)
plt.show()

fig, ax = plt.subplots(1)
ax.plot(ptsmine.real[:-1], ptsmine.imag[:-1] , '-o')
ax.set_aspect(1)
plt.show()

fig, ax = plt.subplots(1)
ax.plot(ptsmine2.real[:-1], ptsmine2.imag[:-1] , '-o')
ax.set_aspect(1)
plt.show()

