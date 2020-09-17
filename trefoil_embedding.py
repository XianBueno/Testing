# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:06:09 2020

@author: Xian
"""

import numpy as np
import matplotlib.pyplot as plt
import sample
from diffusionMap import DiffusionMap
import time

from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

pts = sample.trefoil(N=500)

fig = plt.figure(figsize=plt.figaspect(0.5)*2.5)
ax = fig.add_subplot(projection='3d')
cmap = plt.get_cmap('hsv')
ax.scatter(pts[:,1], pts[:,2], pts[:,3],c=pts[:,0],cmap=cmap,alpha=1)
ax.azim = 50
ax.elev = 50
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

plt.figure()
cmap = plt.get_cmap('hsv')
plt.scatter(pts[:,1],pts[:,2],c=pts[:,0],marker='.',cmap=cmap)
plt.xlabel('x')
plt.ylabel('y')
plt.title('PCA projection of Trefoil')
plt.axis('equal')
plt.tight_layout()
plt.show()

mfld = DiffusionMap(pts[:,1:])
mfld.train(eps=.1,p=3)

plt.figure()
cmap = plt.get_cmap('hsv')
plt.scatter(mfld.pts_dfm[:,1],mfld.pts_dfm[:,2],c=pts[:,0],marker='.',cmap=cmap)
plt.xlabel('1st diffusion coordinate $\Psi_1$')
plt.ylabel('2nd diffusion coordinate $\Psi_2$')
plt.title('2D diffusion maps plot')
plt.axis('equal')
plt.tight_layout()
plt.show()