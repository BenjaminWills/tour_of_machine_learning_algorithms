# Principal Component Analysis (PCA)

PCA is a way for us to reduce the dimensionality of our data so that the most important information is conserved, in general if we have $n$ features we can reduce the features down using PCA to any number less than or equal to $n$. But how do we do this?

Suppose we have a set of $m$ dimensional datapoints $\bold{x}_i$ and we want to find a representation of these features that is $d$ dimensional where $d \leq m$, we want these $d$ dimensional features to retain as much information as is possible. So what does it even mean to reduce the dimensionality? It corresponds to projection onto a line or plane, a projection of a vector $\bold{x}$ onto a unit vector $\bold{u}$ is defined as $\text{proj}(\bold{x})_{\bold{u}} = (\bold{x}\bold{u}^T)\bold{u}$ where $\bold{x}\bold{u}^T$ can be viewed as the similarity between $\bold{x}$ and $\bold{u}$, thus we want to make that quantity as large as possible (which is possible when $\bold{x}$ is paralell to $\bold{u}$). Thus we want to define a line with direction $\bold{u}$ such that each feature vector is as similar as possible to it, and project the feature vectors onto it, we can frame this as an optimisation problem

$$
\max \sum_{i=1}^n (\bold{x}_i\bold{u}^T)^2 \\
\text{subject to } \bold{u}\bold{u}^T = 1 \text{ (unit vector))}
$$

We can rewrite our objective function as follows:

$$
\sum_{i=1}^n (\bold{x}_i^T\bold{u})^2 =\\
\sum_{i=1}^n (\bold{x}_i^T\bold{u})(\bold{x}_i^T\bold{u}) \text{ (The dot product is symmetric)}=\\ 
\sum_{i=1}^n\bold{u}^T\bold{x}_i\bold{x}_i^T\bold{u} =\\ 
\bold{u}^T(\sum_{i=1}^n\bold{x}_i\bold{x}_i)\bold{u} =\\
\bold{u}^TC\bold{u}
$$

Where $C$ is known as the `covariance matrix` and is defined as $C = \frac{1}{n}\sum_{i=1}^n \bold{x}_i\bold{x}_i^T$.

We know that $C$ is a positive definite matrix, and is thus convex, since all of it's elemets must be greater than zero, this means that we can find a global minimum using the `lagrange multiplier` technique, as follows

We introduce a `lagrange paramater` $\lambda$ that defines our lagrange equation:

$$
L(\bold{u}, C, \lambda) =\bold{u}^TC\bold{u}-\lambda(\bold{u}\bold{u}^T - 1) \\
\nabla{L} = C\bold{u} - \lambda\bold{u} \implies \\
C\bold{u} = \lambda\bold{u} \text{ when } \nabla{L} = 0
$$

Which is simply the equation for the eigenvalues of $C$. Note that $C_{ij} = \frac{1}{n}\text{Cov}(\bold{x}_i, \bold{x}_j)$, calculating covariances makes sense as we want to find a line that contains the linear combinations of variables that lead to the most variance in the data and thus carry the most information aobut the data!

Note that $C$ is also symmetric, since the covariance is symmetric we know that the eigenvalues of $C$, $\bold{u}_\lambda$ are orthogonal, so $\bold{u}_\lambda\bold{u}_\lambda^T = 0$ thus it is true that:

$$
C\bold{u} = \lambda\bold{u} \implies \bold{u}^TC\bold{u} = \lambda
$$

This means that the eigenvalue of the covariance is the exact same as the measure of similairty to the eigenvector $\bold{u}$ that we saw before! So we have an interperetation of ranking of how important our principal components are!

So, given a principle direction $\bold{u}$ (which is an eigenvector of $C$) we have an order of imporance $\lambda$ (the eigenvalue corresponding to the principle direction) that allows us to score how much the principle components impact the variance.

So given a principle direction and a datapoint, we can project the datapoint onto the line of the principle direction to get the components of our vectors :), and we can choose up to $n$ eigenvectors to do this with!

So given a principle component $\bold{u}$ (unit vector) and a datapoint $\bold{x}$ we project $\bold{x}$ onto it by finding the inner product between the two, thus the principle component corresponding to $\bold{x}$ has a value of $\bold{x} . \bold{u}$.

In the case when we have a matrix of data $X$ where rows are datapoints, we calcualte it's covariance matrix $C$ and find it's eigenvectors $\bold{u}_{\lambda_i}$ then we can calculate all of the principle components by multiplying our data and our eigenvector which gives $\text{PC}_{\lambda_i} = X \bold{u}_{\lambda_i}$, so if we have a matrix $U = (\bold{u}_{\lambda_0}, \dots, \bold{u}_{\lambda_n})$ we can define ALL of our principal components as $\text{PC} = XU$.