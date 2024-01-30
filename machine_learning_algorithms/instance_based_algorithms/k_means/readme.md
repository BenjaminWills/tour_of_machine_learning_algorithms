# K nearest neighbours

K nearest neighbours is a really simple algorithm, it is essentially the sheep algorithm where any new data point will follow a herd of size $k$.

## Maths behind it

Suppose our training data is a set of data $X \in \mathbb{R}^{m \times n}$ i.e $m$ datapoints with $n$ features each where the $n$th feature is the class of the datapoint. The algorithm works as follows:

1. Choose a value for $k$
2. Input a new datapoint $\bold{x} \in \mathbb{R}^n$
3. Find the $k$ nearest points to $\bold{x}$ in the training set, note that nearest in this case is euclidian distance but can be a different metric!
4. Of the $k$ nearest neighbours to $\bold{x}$ in the training set, find the most common class among them. Then assign that class to $\bold{x}$.

Thus the point $\bold{x}$ defines it's class based on it's neighbours and follows the most frequent class.