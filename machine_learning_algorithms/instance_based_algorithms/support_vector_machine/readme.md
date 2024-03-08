# Support Vector Machines (SVM)

SVM's are used for classificaiton tasks, their goal is to define a hyperplane between two sets of features to identify their classes. The way this works is by creating a plane with a buffer zone, with the aim to make the buffer zone as large as is possible such that the plane is as far from each class as possible whilst still separating them, points on the buffer zone are called `support vectors` as they define the orientation of the plane.

## The maths

Mathematically we can define a plane in any dimension by using an orthogonal vector! This is a vector, $\bold{v}$, such that for any point on the plane (if it goes through the origin), $\bold{u}$ then $\bold{v}.\bold{u}=0 \ \forall \ \bold{u} \text{ in the plane}$. For planes that do not pass through the origin we can write $\bold{v}.\bold{u}=\bold{b} \ \forall \ \bold{u} \text{ in the plane}$ where $\bold{b}$ is just any offset vector of the same dimension.

So we know the problem statement of SVM, we want to find a plane such that all points are a maximal distance from it (i.e it perfectly separates the data), so we create a region around the plane with a width of $w$ we call the distance from the central plane to the support vectors the `margin`. But strangely we want to maximise the distance away from the closest points.

Knowing the problem statement allows us to express SVM in the context of an optimisation problem. We define 3 hyperplanes, and the paramaters that make them up, firstly we have the weights $\bold{w}$, these form the perpendicular vector to the plane and thus it's orientation, next we have the biases $\bold{b}$ that offset the plane from the origin, so our three planes are defined as follows:

$$
\text{for any point } \bold{x} \text{ we have three equations:} \\
(1) \ \bold{w} . \bold{x}  + \bold{b}= 0 \\
(2) \ \bold{w} . \bold{x}  + \bold{b}= 1 \\
(3) \ \bold{w} . \bold{x}  + \bold{b}= -1 \\
$$

The first is the base plane, the second is the upper hyperplane and the third is the lower hyperplane. These planes form a sort of margin with width $w = w_+ + w_-$ where $w_+$ is the shortest distance from the base plane to the closest positive class point, and $w_-$ is the shortest distance from the base plane to the closest negative class point. So the only thing that impacts this classification problem is the support vectors! The others don't matter. Our intention is to maximise $w$.

So how can we mathematically calculate this margin? The perpendicular distance between the base plane (1) and the positive plane (2) is $\frac{1}{|\bold{w}|}$, thus the total difference between the supporting hyperplanes is $\frac{2}{|\bold{w}|}$, thus in order to **maximise** the margin, we need to **minimise** the magnitude of the weights! We also have the condition that there are **no** points that can exist between (2) and (3).

We know that we can define for two classes $y_i \in \{-1,+1\}$ the hyperplanes (2) and (3):

$$
\bold{x}.\bold{w} + b \geq 1 \text{ Any point that exists in this region would be classified as +1} \\
\bold{x}.\bold{w} + b \leq -1 \text{ Any point that exists in this region would be classified as -1}\\
-1 < \bold{x}.\bold{w} + b < 1 \text{ This is the region between the hyperplanes}\\
$$

We can combine those two first conditions as $y_i(\bold{x}.\bold{w}) + \bold{b} \geq 1$

So finally we can frame our optimisation problem, we want to minimise $|\bold{w}|$ such that no points exist within the decision boundary. That is to say we have an objective function $f = \frac{|\bold{w}|^2}{2}$ (we square it to make it a convex function that is way easier to minimse as it takes the form of a parabola which always has a global minimum, further a global minimum) that is subject to the constraint function $g(x) = y_i(\bold{x}_i.\bold{w}) + \bold{b} = 1$ i.e points are on the decision boundary. We can solve this using the lagrangian method.

Our lagrangian for this problem takes the form of 

$$
L = \frac{|\bold{w}|^2}{2} + \sum_{i=1}^{d}\lambda_i(y_i(\bold{x}_i.\bold{w}) + b) - \sum_{i=1}^{d}\lambda_i
$$

By minimising $L$ we can find the optimal solution to this problem, note that $\lambda_i$ is the largrange multiplier for the $i$'th constraint and $d$ is the number of datapoints in the dataset. Thus taking the gradient of $L$ and setting it $0$ will give us our solution! Doing that leads to the following:

$$
\bold{w} = \sum_{i=1}^{d}\lambda_iy_i\bold{x}_i, \ \sum_{i=1}^{d}\lambda_iy_i = 0
$$

We can subsitute these relations into the original lagrangian $L$ to attain the **dual** problem, which means we need only optimise w.r.t $\lambda_i$ i.e the lagrange multipliers! Also we will find that most lagrange multipliers will turn out to be $0$! The ones that are non-zero are support vectors!

Thus our problem can be simplified to just a function of the lagrange multipliers:

$$
L = \frac{|\bold{w}|^2}{2} + \sum_{i=1}^{d}\lambda_i(y_i(\bold{x}_i.\bold{w}) + b) - \sum_{i=1}^{d}\lambda_i \\
= \frac{\bold{w}\bold{w}^T}{2} - \bold{w}.\sum_{i=1}^{d}\lambda_iy_i\bold{x}_i - \sum_{i=1}^{d}\lambda_i \\
= \frac{\bold{w}\bold{w}^T}{2} - \bold{w}.\bold{w}^T + \sum_{i=1}^{d}\lambda_i \\
= \sum_{i=1}^{d}\lambda_i - \frac{\bold{w}\bold{w}^T}{2}\\
= \sum_{i=1}^{d}\lambda_i - \frac{1}{2} \sum_{i=1}^{d} \sum_{j=1}^{d}\lambda_i \lambda_ jy_iy_j(\bold{x}_i.\bold{x}_j)
$$

Hence we can summarise this by writing the optimisation problem as follows:

$$
\max_\lambda L_D = \sum_{i=1}^{d}\lambda_i - \frac{1}{2} \sum_{i=1}^{d} \sum_{j=1}^{d}\lambda_i \lambda_ jy_iy_j(\bold{x}_i.\bold{x}_j) \\
\text{with the linear constraints: } \lambda_i \geq 0, \sum_{i=1}^{d}\lambda_iy_i = 0
$$

Thus we can find our lagrange multipliers $\bold{\lambda}$ and thus our weights. To find our biases $b_i$ for each datapoint:

$$
y_i(\bold{w}^T\bold{x} + b_i) = 1 \\
b_i = y_i - \bold{w}^T\bold{x}_i \\
b = \frac{\sum_{i=1}^dy_i - \bold{w}^T\bold{x}_i}{d}
$$