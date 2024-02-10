# K means clustering

K means clustering is an unsupervised learning algorithm that allows the data to be grouped into clusters based on k `centroids`, which are points at the center of each cluster. K means clustering has the following goals

1. **All the data points in a cluster should be similar to each other**: By some metric all points within a cluster must be similar, otherwise the interpretation of the cluster is meaningless.
2. **The data points from different clusters should be as different as possible**: So that the clusters can be as destinct as possible and have no overlap in interperetation.

The main objective of the K-Means algorithm is to minimize the sum of distances between the points and their respective cluster centroid.

## How can we assess how `good` a set of clusters is?

### Inertia

Inertia evaluates how similar each datapoint in a cluster is by calculating the distance between a centroid of a cluster and all of the points within the cluster. We want this value to be as small as possible as this means that all of the points are close to the centroid and thus are similar. So if we have $n$ points ($\bold{x}_i$) in a cluster with a centroid of $\bold{x}_c$ then the inertia is calculated as:

$$
\text{Inertia} = \sum_{i = 1}^{n}|\bold{x}_c - \bold{x}_i|
$$

Where $|\bold{x}_c - \bold{x}_i|$ is known as the intracluster distance between point $i$ and cluster $c$.

### The Dunn index

Inertia only checks if points in clusters are similar to one another, thus it does not check that the clusters are distinct. The Dunn index takes into account the distance between the centroids of the clusters, hence checking that the clusters are distinct. We define the term intercluster distance as the distance between two clusters.

$$
\text{Dunn Index} = \frac{\min{(\text{intercluster distance})}}{\max({\text{intracluster distance})}}
$$

This is just the ratio of the smallest intercluster distance to the largest, we want to maximise the dunn index as we want the smallest intercluster differnece to be large (as the clusters should be distinct and thus far apart by some distance metric) and the largest intracluster distance to be small as then clusters will tend to be compact.

## The algorithm

Suppose we have $n$ datapoints with $m$ features such that a datapooint $\bold{x} \in \mathbb{R}^m$. Also we have set a maximum iteration count of $N$ and $i$ is the number of iterations that have been tracked.

1. Pick $k$ clusters where $n \geq k$
2. Choose $k$ data points randomly to be the `centroids` in our algorithm
3. Find the closest data points to each centroid
4. Calculate the Inertia and Dunn index of the data to keep track of metrics
5. Stopping criteria:
   1. Centroids of newly formed clusters do not change
   2. Points remain in the same cluster
   3. Maximum number of iterations is reached ($N$)
6. Find the average value of the points surrounding each centroid and set that to be the new value
7. $i = i + 1$ and go back to step 3.