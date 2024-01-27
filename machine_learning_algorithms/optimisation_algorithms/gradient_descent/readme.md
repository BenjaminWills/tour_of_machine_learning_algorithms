# Gradient descent

Gradient descent is a numerical method of optimising a function which relies on the fact that the gradient vector of some funciton is **always** the steepest point for ascent or descent.

## Intuition

TO FILL IN

## Mathematical formulation

Given a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ and an initial starting point $\bold{x}_0 \in \mathbb{R}^n$, the steepest point of descent at any point $\bold{x}$ on $f$ is $\nabla f(\bold{x})$, so we can travel down that vector at each point. So the iterative method is

$$\bold{x}_{i+1} = \bold{x}_{i} - \alpha \nabla f(\bold{x_i})$$

Where $\alpha$ is the `learning rate` which is how much of a step we take at each descent point