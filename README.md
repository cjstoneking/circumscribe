# circumscribe
visualize complex classifier outputs, simply and intuitively

This is a small Python library for visualizing the output of classifiers on points in a 2D plane. It is intended for the following type of situation: we have a classification algorithm, and we want to visualize its classifications on a dataset, in a way that is visually clean and intuitively understandable for non-experts. In particular, we want to be able to do this even if the classification problem is complicated (e.g. if there are >2 classes, or irregular classification boundaries, or single classes occupy multiple disjoint regions of space). In this situation, the standard visualization approach (matplotlib contour/contourf) tends to produce overly complicated plots that are difficult for non-experts to interpret.

The approach taken by this library is to find a set of closed contours, labeled according to the different classes, such that each point is enclosed by a contour of the correct class. This is subtly different from other visualization approaches: we are not trying to visualize the classifier's prediction at each point in 2D space, instead we are focusing on visualizing how it labels the individual data points, and ignoring the space that the points are embedded in. This results in a visually simpler plot, which is also more intuitive for non-experts to use (because most people tend to think of a classification algorithm as something that operates on data points, not on all of space).

Another way of thinking about this approach is: we are trying to visualize the classifier output the way most humans would, by circling groups of points that belong to the same class. We are not trying to color in all regions of space, or draw complicated boundaries that handle all of space (which is the contour/contourf approach).

It's important to note that circumscribe contours do not give a faithful representation of the classifier decision boundaries. They do give a faithful representation of the classes that the classifier assigns to the data points.

The contours are computed according to rules that are intended to yield a visually simple plot:

- the total number of contours is kept to a minimum
- contours are kept "as convex as possible": sections are convex by default, concavities only introduced when necessary
- smoothing is applied to avoid jaggedness

Example:

![circumscribe_demo_nonconvex_01](figures/circumscribe_demo_nonconvex_01.png?raw=true "Example of slightly nonconvex contours")


