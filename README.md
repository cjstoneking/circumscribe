# circumscribe
visualize complex classifier outputs, simply and intuitively

This is a small Python library for visualizing the output of classifiers on points in a 2D plane. It is intended for the following type of situation: we have a classification algorithm, and we want to visualize its classifications on a dataset, in a way that is visually clean and intuitively understandable for non-experts. In particular, we want to be able to do this even if the classification problem is complicated (e.g. if there are >2 classes, or irregular classification boundaries, or single classes occupy multiple disjoint regions of space). 
In this situation, the standard visualization approach

The approach taken by this library is to find a set of closed contours, labeled according to the different classes, such that each point is enclosed by a contour of the correct class. This is subtly different from other visualization approaches: we are not trying to visualize the classifier's prediction at each point in 2D space, instead we are focusing on visualizing how it labels the individual data points, and ignoring the rest of space. This results in a visually simpler plot, which is also more intuitive for non-experts to use (because most people will think of a classification algorithm as something that operates on data points, not on all space).  
