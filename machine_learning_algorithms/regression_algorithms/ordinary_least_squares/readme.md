# Ordinary least squares regression (OLSR)

Suppose we have $n$ observations of some $p$ dimensional set of variables $\bold{x}$ that have a paired variable $y$, $(\bold{x}, y)$. The idea of OLSR is to find a linear formula to define $y$. You may be familiar with the classic $y = mx + c$ formula in 2 dimensions. 

As an example we can predict a persons weight given their height. In this example $\bold{x}$ is a 1 dimensional variable that represents the persons height, and $y$ is the persons height.

## Mathematical formulation

We can formulate this linear equation for a $p$ dimensional independent variable. Suppose we have a training set of $n$ pairs of variables $(\bold{x}, y)$ where $x\in \mathbb{R}^p$ and $y\in \mathbb{R}$. We can define the matrix of independent variables $X \in \mathbb{R}^{n \times p}$ and some offsets $\bold{\epsilon} \in \mathbb{R}^n$ and some weights $\bold{\beta}\in \mathbb{R}^p$, then we can express the set of $n$ independent variables as the following:

$$\bold{y}_{pred} =X\bold{\beta} + \bold{\epsilon}$$

The idea of OLSR is to adjust $\bold{\beta},\bold{\epsilon}$ such that $\bold{y}_{pred}$ is as close as possible to $\bold{y}$, to measure this "closeness" of prediction to real, we consider the difference of our predictions to the real values: $\bold{y} - \bold{y}_{pred}$. When we have loads of variables we want to minimise the average difference: $S(\bold{y}, \bold{\beta}) = |\bold{\epsilon}|^2 = |\bold{y} - X\bold{\beta}|^2 = \bold{\epsilon}\bold{\epsilon}^T$.

In 2D this looks like fitting a line as closely as possible to the data, creating a line of best fit.

### Minimising $S$

We can frame the problem now entirely through the lense of optimising w.r.t one variable, suppose we have $n$ dependent variables, $(y_1, \dots, y_n)$, and $n$ sets of $m$ dimensional independent variables $(\bold{x}_1, \dots, \bold{x}_n)$. Define a set of paramaters $\bold{\beta}\in \mathbb{R}^{m+1}$ and an estimator $\hat{y}(\bold{x}_j) = \beta_0 + \sum_{i=1}^{n}\beta_ix_i^{(j)} = \bold{\beta}.\bold{x}_j$ where we have modifed $\bold{x}$ to be $m+1$ dimensional with $x_0 = 1$ to ensure that $\beta_0$ occurs in the estimator $\hat{y}$. So now we can define $\bold{\hat{y}} = A\bold{\beta}$ where $A \in \mathbb{R}^{n \times (m+1)}$ and $A_{ij} = x_i^{(j)}$. The average error is then defined as $\frac{1}{n}|\bold{y} - \bold{\hat{y }}|^2$. So we aim to find a vector $\bold{\hat{y}}$ that will minimise this sum, logically that will be a vector that is perpendicular to $\bold{y} - \bold{\hat{y }}$ as that will be the closest vector to it. $\bold{\hat{y}}.(\bold{y} - \bold{\hat{y }}) = (A\beta)(\bold{y} - A \beta) = 0$. This equation simplifies down to $(A^TA)^{-1}A^T\bold{y} = \beta$. This version of $\beta$ successfully. Thus we have a general formula to solve regression!