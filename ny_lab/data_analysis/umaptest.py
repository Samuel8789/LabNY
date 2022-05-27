# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:22:50 2022

@author: sp3660
"""
import pickle
import umap
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import numpy as np
import umap.plot
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from  sklearn.decomposition import PCA

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


penguins = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
penguins.head()






direct=r'K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPKG\data\JesusRuns'
inter=glob.glob(direct+'\*int*')
allcells=glob.glob(direct+'\*All_jesus*')
pyr=glob.glob(direct+'\*pyr_jesus*')







with open(inter[0], 'rb') as file:
  intdfdt=  pickle.load(file)
   
  
act=intdfdt[4]['Raster']  
pca = PCA(n_components=3)
res=pca.fit_transform(act.T)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

ax.scatter(res[:,0]
           , res[:,1]
           , s = 50)
ax.grid()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(res[:,0], res[:,1], res[:,2])



neurosn=intdfdt[4]['Ensembles']['EnsembleNeurons']

reducer = umap.UMAP()
embedding_vectors = reducer.fit_transform(act.T)
embedding_cells = reducer.fit_transform(act)

embedding_vectors.shape
embedding_cells.shape

plt.scatter(
    embedding_vectors[:, 0],
    embedding_vectors[:, 1],
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of Drifitng Gratings')

plt.figure()
plt.scatter(
    embedding_cells[:, 0],
    embedding_cells[:, 1],
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of Drifitng Gratings')



mapper = umap.UMAP(metric='cosine', random_state=42, low_memory=True).fit(act)
umap.plot.points(mapper, values=np.arange(100000), theme='viridis')
