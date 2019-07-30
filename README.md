# circumscribe
visualize complex classifier outputs, in a clean and polished manner

This is a small Python library for visualizing the output of classifiers on points in a 2D plane. It is intended for the situation where we have used a black-box classifier (e.g. lightGBM, xgboost, random forest...), so although we have predicted labels for the data points, we don't have a simple representation of the decision boundary. In this case, the standard approach for visualizing the classifier output is to apply nearest-neighbors classification to all points in space, then plot the resulting contours directly (e.g. with matplotlib contour/contourf). This will often result in highly irregular contours that are not visually clean, even when smoothing is applied.

The circumscribe function solves this problem by using contours that are forced to be as convex as possible, and these tend to look substantially neater:

![standard_vs_circumscribe_01](figures/standard_vs_circumscribe_01.png?raw=true "circumscribe vs mpl contour plot")



Further details:

The approach taken by this library is to find a set of closed contours, labeled according to the different classes, such that each point is enclosed by a contour of the correct class. It's important to note that circumscribe contours do not necessarily give a faithful representation of the classifier decision boundaries. They do give a faithful representation of the classes that the classifier assigns to the data points.

The contours are computed according to rules that are intended to yield a visually simple plot:

- the total number of contours is kept to a minimum
- contours are kept "as convex as possible": sections are convex by default, concavities only introduced when necessary
- smoothing is applied to avoid jaggedness

![circumscribe_demo_nonconvex_01](figures/circumscribe_demo_nonconvex_01.png?raw=true "Example of slightly nonconvex contours")




