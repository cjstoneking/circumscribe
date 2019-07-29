# circumscribe
visualize complex classifier outputs, in a clean and polished manner

This is a small Python library for visualizing the output of classifiers on points in a 2D plane, in a way that is visually cleaner than the standard approach (applying nearest-neighbors to all points in space, then using matplotlib contour/contourf):

![circumscribe_vs_mpl_contour_03](figures/circumscribe_vs_mpl_contour_03.png?raw=true "circumscribe vs mpl contour plot")

The circumscribe function uses contours that are forced to be as convex as possible, and these tend to look substantially tidier than the equivalent contours produced by the matplotlib contour function (even when smoothing is applied).

Further details:

The approach taken by this library is to find a set of closed contours, labeled according to the different classes, such that each point is enclosed by a contour of the correct class. This is subtly different from other visualization approaches: we are not trying to visualize the classifier's prediction at each point in 2D space, instead we are focusing on visualizing how it labels the individual data points, and ignoring the space that the points are embedded in. This results in a visually simpler plot, which is also more intuitive for non-experts to use (because most people tend to think of a classification algorithm as something that operates on data points, not on all of space).

Another way of thinking about this approach is: we are trying to visualize the classifier output the way most humans would, by circling groups of points that belong to the same class. We are not trying to color in all regions of space, or draw complicated boundaries that handle all of space (which is the contour/contourf approach).

It's important to note that circumscribe contours do not give a faithful representation of the classifier decision boundaries. They do give a faithful representation of the classes that the classifier assigns to the data points.

The contours are computed according to rules that are intended to yield a visually simple plot:

- the total number of contours is kept to a minimum
- contours are kept "as convex as possible": sections are convex by default, concavities only introduced when necessary
- smoothing is applied to avoid jaggedness

Examples:

![circumscribe_demo_nonconvex_01](figures/circumscribe_demo_nonconvex_01.png?raw=true "Example of slightly nonconvex contours")



![circumscribe_vs_mpl_contour_01](figures/circumscribe_vs_mpl_contour_01.png?raw=true "circumscribe vs mpl contour plot")
