# MIT License
#
# Copyright (c) 2019 Colin James Stoneking

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


#This function is a wrapper that enables matplotlib contour/contourf functions to be used
#to produce similar plots as circumscribe

#Used for comparisons

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def mpl_contourplot(ax, x, y, labels, xvals=np.arange(0,100,1), yvals=np.arange(0,100,1), function = 'contour', sigma=1, min_dist=10):
    
    X, Y = np.meshgrid(xvals, yvals)
    unique_labels = np.sort(np.unique(labels))
    
    canvas = np.zeros([len(unique_labels), X.shape[0], X.shape[1]])
    label_to_ind = {}
    for n, lab in enumerate(unique_labels):
        label_to_ind[lab] = n
    
    canvas_to_space_x = lambda xc : xc/float(X.shape[0])*(xvals[-1]  - xvals[0]) + xvals[0]
    canvas_to_space_y = lambda yc : yc/float(X.shape[1])*(yvals[-1]  - yvals[0]) + yvals[0]
    
    #do nearest_neighbor classification of each pixel
    for xc in range(canvas.shape[1]):
        for yc in range(canvas.shape[2]):
            
            xp = canvas_to_space_x(xc)
            yp = canvas_to_space_y(yc)
            
            d = np.sqrt((x - xp)**2 + (y - yp)**2)
            
            if(np.min(d)<= min_dist):
                nn_ind = np.argsort(d)[0]
                nn_lab = labels[nn_ind]#nearest-neighbor based label
                canvas[label_to_ind[nn_lab],xc,yc] = 1
    
    #now each "layer" of the canvas is a binary image
    #gaussian-smooth them
    for z in range(canvas.shape[0]):
        canvas[z,:,:] = gaussian_filter(canvas[z,:,:], sigma=sigma, mode='constant', cval=0)
        
    #now combine canvases into single height matrix for contour/contourf
    Z = np.zeros(X.shape)
    for z in range(canvas.shape[0]):
        Z[canvas[z,:,:]>0.5]=(z+1)*10
    
    if(function=='contour'):
        ax.contour(Y,X,Z)
    elif(function=='contourf'):
        ax.contourf(Y,X,Z)
