"""
hall_probes_to_fake_data.py

author: Colin Clement
Date: 2016-2-10

This scipt was used to identify the shape of the hallprobes in an optical image
of a hall probe setup. The raw mask image is saved ini data/hallprovemask.npy.
This mask was then interpolated to live on the same
grid as the data in im_HgTe_Inv_g_n1..., expanded to fit in the bottom, and
saved in data/hallprobe_interpolated.npy

"""

import numpy as np
import scipy as sp
import os
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from pysquid.util.helpers import loadpng, plotrgba
from skimage.transform import rotate
from scipy.interpolate import griddata
from scipy.io import loadmat

loc = os.path.dirname(os.path.realpath(__file__))

rgba = rotate(loadpng(os.path.join(loc, '../../../katja/MicroscopePicture.png')), 1)
#immask = (rgba[:,:,1]<np.mean(rgba[:,:,1]))*(rgba[:,:,0]<np.mean(rgba[:,:,0]))
immask = (rgba[:,:,1]<0.0075)*(rgba[:,:,0]<0.00125)
h, w, _ = rgba.shape

closed = immask[10:h//2,75:-70]
#plt.matshow(closed, cmap='Greys')
labelled, num = ndi.label(closed)

bins = np.arange(num+1)
hist, bins = np.histogram(labelled.ravel(), bins)
order = np.argsort(hist)[::-1]

probelist = list(bins[order])
probelist.pop(0) #remove background

left = 1*(labelled == 1)
right = 1*(labelled==13)
mask = ndi.binary_fill_holes(left + right)
opened_mask = ndi.binary_opening(mask, iterations=2)
h = np.cumsum(opened_mask.sum(1)==0) == 1
v = opened_mask.sum(0)==0
bridge = np.outer(h,v)
bridged_mask = ndi.binary_dilation(ndi.binary_closing(opened_mask+
                                                      bridge, 
                                                      iterations=3))

#plt.matshow(bridged_mask, cmap='Greys')
ly, lx = bridged_mask.shape
padded = np.zeros((ly+200, lx))
padded[:ly,:] = bridged_mask
np.save(os.path.join(loc, 'fake_data_hallprobemask.npy'), padded[3:,5:-5])

mask = padded[3:,5:-5]

sample = rgba[230:290,335:435,1].copy()
sample[:,12] = 0
sample[:,-14] = 0
cut = sample[:,13:-14]
#plt.matshow(cut)
scope_scale = 50./(cut.shape[1]) #micrometers

data = loadmat(os.path.join(loc, '../../../katja/im_HgTe_Inv_g_n1_061112_0447.mat'))
scan = data['scans'][0][0][0][::-1,:,2]
ly, lx = scan.shape
imgspacing = [.16, .73]

mly, mlx = mask.shape
x, y = np.arange(mlx)*scope_scale, np.arange(mly)*scope_scale
xg, yg = np.meshgrid(x, y)

datax = np.arange(0, mlx*scope_scale, imgspacing[1])
datay = np.arange(0, mly*scope_scale, imgspacing[0])
dxg, dyg = np.meshgrid(datax, datay)

xi = np.c_[dxg.ravel().T, dyg.ravel().T]
points = np.c_[xg.ravel().T, yg.ravel().T]
interp = griddata(points, 1*mask.ravel(), xi, method='nearest', fill_value=0)

#Alignment of scan image with interpolated mask
xcorner, ycorner = 224, 835

np.save(os.path.join(loc, 'fake_data_hallprobe_interpolated.npy'),
        interp.reshape(len(datay), -1))


