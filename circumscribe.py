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


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def _get_boundary_points(x, y, labels, min_dist = 10, margins = [10, 10], n_pixels=[200, 200],  sigma = 1, bval = 0.5, connectivity=4):
    
    unique_labels = np.sort(np.unique(labels))
    canvas = np.zeros([len(unique_labels), n_pixels[0], n_pixels[1]])
    label_to_ind = {}
    for n, lab in enumerate(unique_labels):
        label_to_ind[lab] = n
    
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    xmarg = margins[0]
    ymarg = margins[1]
    
    canvas_to_space_x = lambda xc : xc/float(n_pixels[0])*(xmax + xmarg  - (xmin - xmarg)) + (xmin - xmarg)
    canvas_to_space_y = lambda yc : yc/float(n_pixels[1])*(ymax + ymarg  - (ymin - ymarg)) + (ymin - ymarg)
    
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
        
    #now get boundary points
    boundary_points = {}
    
    if(connectivity==4):
        neighbor_offsets = np.array([[-1,0], [0,1], [1,0], [0,-1]])
    elif(connectivity==8):
        neighbor_offsets = np.array([[-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1]])
    else:
        print("Connectivity must be 4 or 8")
    
    for lab in unique_labels:
        layer = canvas[label_to_ind[lab],:,:]
        temp_list = []
        for xc in range(layer.shape[0]):
            for yc in range(layer.shape[1]):
                
                if(layer[xc, yc] >= bval):
                
                    found_border = False
                    for n in range(neighbor_offsets.shape[0]):
                        xn = xc + neighbor_offsets[n,0]
                        yn = yc + neighbor_offsets[n,1]
                        if(xn not in range(canvas.shape[1]) or yn not in range(canvas.shape[1]) or layer[xn,yn] < bval):
                            found_border = True
                            break
                    if(found_border):
                        temp_list.append([canvas_to_space_x(xc), canvas_to_space_y(yc)])
        boundary_points[lab] = np.array(temp_list)
    return boundary_points
    
    
def _get_contours_from_boundary_points(B, cutoff_dist = 1):
    
    
    S = {}
    P = {}
    H = {}
    #store convex hulls here
    
    all_x = []
    all_y = []
    all_labels = []
    
    for key, b in B.items():
        all_labels = all_labels + [key]*b.shape[0]
        all_x = all_x + list(b[:,0])
        all_y = all_y + list(b[:,1])
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    for key, b in B.items():
        label = np.zeros(b.shape[0])
        added_neighbors = np.zeros(b.shape[0])
        
        d = np.zeros([b.shape[0], b.shape[0]])
        for i in range(b.shape[0]):
            for j in range(b.shape[0]):
                d[i,j] = np.sum(np.square(b[i,:] - b[j,:]))
        current_label_value = 1
        for i in range(b.shape[0]):
            if(label[i]==0):
                #if the point is not yet labeled:
                
                #1. label the point and its neighbors
                label[d[i,:] < cutoff_dist] = current_label_value
                added_neighbors[i] = 1
                
                #2. propagate the label from neighbors to their neighbors, etc.
                while(True):
                    do_another_pass = False
                    for j in range(b.shape[0]):
                        if(label[j]==current_label_value and not added_neighbors[j]):
                            label[d[j,:] < cutoff_dist] = current_label_value
                            added_neighbors[j] = 1
                            do_another_pass = True
                    if(not do_another_pass):
                        break
                        
                #3. step to next label value
                current_label_value = current_label_value + 1
        unique_labels = np.unique(label)
        contours = []
        points = []
        hulls = []
        for u in unique_labels:
            all_points = b[label==u,:]

            hull = ConvexHull(all_points)
            hull_indices = list(hull.vertices) 
            #this is just a list of integers
            hull_indices.append(hull_indices[0])
            #repeat first point at end so we have a closed contour
            #hull_points = all_points[hull_indices,:]
            #hull_points = np.concatenate([hull_points, hull_points[0,:][np.newaxis,:]], axis=0)
  

            hull_points = all_points[hull_indices,:]
                
            
            contours.append(hull_points)
            points.append(all_points)
            hulls.append(hull)
        S[key] = contours
        P[key] = points
        H[key] = hulls
        
    #now we have initialized to a set of convex contours
    #next, check for intersections
    
    keylist = list(S.keys())
    
    for k1 in range(len(keylist)):
        for k2 in range(k1+1, len(keylist)):
            
 
            contours1 = S[keylist[k1]]
            contours2 = S[keylist[k2]]
            allpoints1 = P[keylist[k1]]
            allpoints2 = P[keylist[k2]]
            #each of these is a list of multiple closed convex contours
            hulls1 = H[keylist[k1]]
            hulls2 = H[keylist[k2]]
                
            for ci1 in range(len(contours1)):
                for ci2 in range(len(contours2)):
                    c1 = contours1[ci1]
                    c2 = contours2[ci2]
                    #check if these contours intersect
                    #first, do a basic min-max check to see if we can rule out intersection
                        
                    minmaxcheck = lambda c1, c2 :False \
                    or ((np.max(c1[:,0]) <= np.min(c2[:,0])) and (np.max(c1[:,1]) <= np.min(c2[:,1])))\
                    or ((np.max(c2[:,0]) <= np.min(c1[:,0])) and (np.max(c2[:,1]) <= np.min(c1[:,1])))\
                    or ((np.max(c1[:,0]) <= np.min(c2[:,0])) and (np.max(c2[:,1]) <= np.min(c1[:,1])))\
                    or ((np.max(c2[:,0]) <= np.min(c1[:,0])) and (np.max(c1[:,1]) <= np.min(c2[:,1]))) 
                    
                    if(minmaxcheck(c1, c2)):
                        #the contours can't intersect
                        continue
                                                    
                    a = np.pi/4    
                    R = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
                    c1r = np.matmul(c1, R)
                    c2r = np.matmul(c2, R)
                        
                    if(minmaxcheck(c1r, c2r)):
                        #the contours can't intersect
                        continue
                            
                    poly1 = Polygon([(c1[i,0], c1[i,1]) for i in range(c1.shape[0]) ])
                    poly2 = Polygon([(c2[i,0], c2[i,1]) for i in range(c2.shape[0]) ])
                    points1 = [Point(c1[i,0], c1[i,1]) for i in range(c1.shape[0])]
                    points2 = [Point(c2[i,0], c2[i,1]) for i in range(c2.shape[0])]
                    intersect1 = []
                    #points from contour 1 that lie within contour 2
                    intersect2 = []
                        
                    centroid1 = np.mean(c1, axis=0)
                    centroid2 = np.mean(c2, axis=0)
                        
                    for i in range(c1.shape[0]):
                        if(poly2.contains(points1[i])):
                            intersect1.append(c1[i,:])
                                
                    for i in range(c2.shape[0]):
                        if(poly1.contains(points2[i])):
                            intersect2.append(c2[i,:])
                            
                    if(len(intersect1)>0 or len(intersect2)>0):
                        
                        intersect1 = np.array(intersect1)
                        intersect2 = np.array(intersect2)
                        if(intersect1.shape[0] > 0 and intersect2.shape[0] > 0):
                            full_intersect = np.concatenate([intersect1, intersect2], axis=0)
                        elif(intersect1.shape[0] > 0):
                            full_intersect = intersect1
                        else:
                            full_intersect = intersect2
                        min_x = np.min(full_intersect[:,0]) 
                        max_x = np.max(full_intersect[:,0]) 
                        min_y = np.min(full_intersect[:,1]) 
                        max_y = np.max(full_intersect[:,1]) 
                            
                        bounds = np.array([min_x, min_y, max_x, max_y])
                        
                        contours1[ci1] = _get_single_contour(allpoints1[ci1], hulls1[ci1], bounds)
                        contours2[ci2] = _get_single_contour(allpoints2[ci2], hulls2[ci2], bounds)
                #end loop over ci2
            #end loop over ci1
        #end loop over k2
    #end loop over k1
                        
    return S
    
def _get_single_contour(p, hull, bounds):
    
    in_bounds = lambda q : bounds[0] <= q[0] and q[0] <= bounds[2] and bounds[1] <= q[1] and q[1] <= bounds[3]
    
    #hull = ConvexHull(p)
    hull_indices = list(hull.vertices) 
    #this is just a list of integers
    hull_indices.append(hull_indices[0])
    #repeat first point at end so we have a closed contour

    ind_not_in_hull = [j for j in range(p.shape[0]) if j not in hull_indices]
    
    in_hull = np.zeros(p.shape[0])
    for hi in hull_indices:
        in_hull[hi] = 1
    
    D = np.zeros([p.shape[0], p.shape[0]])
    for i in range(p.shape[0]):
        for j in range(p.shape[0]):
            D[i,j] = np.sum(np.square(p[i,:] - p[j,:]))
    
    while(True):
        hull_indices_copy = [hi for hi in hull_indices]
        insert_offset = 0
        #this gets incremented by 1 each time a point is added
        #it ensures that we are adding new points to the hull_indices_copy
        #at the correct locations
        for k in range(len(hull_indices)-1):
            i1 = hull_indices[k]
            i2 = hull_indices[k+1]
            insertion_candidates = []
            scores = []
            for j in range(p.shape[0]):
                if(not in_hull[j] and in_bounds(p[j,:]) and D[i1,j] < D[i1, i2] and D[i2, j] < D[i1, i2]):
                    in_hull[j] = 1
                    hull_indices_copy.insert(k+insert_offset+1, j)
                    insert_offset = insert_offset + 1
                    break
                            
        did_second_stage_insert=False
        if(len(hull_indices_copy) == len(hull_indices)):
            #found nothing more to insert by validity
            if(not all(in_hull)):
                best_j = -1
                best_k = -1
                best_dist = np.inf
                for j in range(p.shape[0]):
                    if(not in_hull[j] and in_bounds(p[j,:])):
                        for k in range(len(hull_indices)-1):
                            i1 = hull_indices[k]
                            d_to_hull = D[:,j]
                            prev_k = k-1
                            if(prev_k <0):
                                prev_k = len(hull_indices)-1
                            post_k = k+1
                            if(post_k > len(hull_indices)-1):
                                post_k = 0
                            prev_dist = D[hull_indices[prev_k],j]
                            post_dist = D[hull_indices[post_k],j]
                            
                            if(best_dist > D[i1,j] and post_dist<prev_dist):
                                best_dist = D[i1,j]
                                best_j = j
                                best_k = k
                #end loop over j
                if(best_j>=0):
                    in_hull[best_j] = 1
                    old_index = hull_indices_copy[best_k+insert_offset+1]
                    #in
                    hull_indices_copy[best_k+insert_offset+1]=best_j
                    #hull_indices_copy.insert(best_k+insert_offset+1, best_j)
                    insert_offset = insert_offset + 1
                    
        if(insert_offset == 0):
            break
        hull_indices = hull_indices_copy
        
        
    hull_points = p[hull_indices,:]
    return hull_points
    
#this is the function intended to be called by user
def circumscribe_contours(x, y, labels, min_dist = 10, margins = [10, 10], n_pixels=[200, 200],  sigma = 1, bval = 0.5, connectivity=4):
    B = _get_boundary_points(x, y, labels, min_dist=min_dist, margins=margins, n_pixels=n_pixels, sigma=sigma, bval=bval, connectivity=connectivity)
    S = _get_contours_from_boundary_points(B)
    return S
    
