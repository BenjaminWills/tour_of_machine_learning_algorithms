# Logistic regression

Logistic regression is the study of using the `logit` funciton to assign probabilities to classes and to assign a `decision boundary` within data.

The logit function is defined as follows for some probability $p \in [0,1]$.

$$\text{logit}(p) = 
\log{\frac{p}{1-p}}$$

Where $\frac{p}{1-p}$ are defined as `odds` corresponding to the probability $p$ as they are the ratio of the chance that an event does happen to the chance that the event does not happen. Thus a logit function is defined as the `log-odds` of a probability $p$, we can find p in terms of the log-odds, lets call them $x$.

$$p(x) = \frac{1}{1+e^{-x}}$$

This function is called a `sigmoid` and has the property that it is always within $0$ and $1$ no matter what value of $x$ we substitute.

Logistic regression states the following, given a set of dependent variables $\bold{x} \in \mathbb{R}^{n+1}$ where $n$ is the number of dependent variables and some weights $\bold{\beta} \in \mathbb{R}^{n+1}$:

$$\log{\frac{p}{1-p}} = \beta_0x_0 + \beta_1x_1 + ... + \beta_nx_n = \sum_{i = 0}^{n}\beta_ix_i = \bold{\beta}.\bold{x}$$ where $x_0 = 1$. 

i.e the logit function is linear in the dependent variables. By this hypothesis we can find a set of weights that will match these log odds directly! So the logistic regression forumla is given by:

$$\mathbb{P}_\beta(y = 1 | \bold{x}) = p(\bold{x}, \bold{\beta}) = \frac{1}{1+e^{-\bold{\beta}.\bold{x}}} \in [0,1]$$

Where $y = 1$ is the probability that the calss is 1.

## Minimising $l$

To minimise l we look at a property called maximum likelyhood estimation. But we can skip all of this work and just look at the cross entropy loss function, since we know that it penalises low certainty in a class which is exactly what we want here, but, as we will shortly see - the maximum likelyhood equation for our set up is exactly the cross entropy loss function.

Suppose that we have $m$ realisations of our dependent variable $\bold{x} \in \mathbb{R}^n$ where n is the number of dependent variables. Then the likelyhood of seeing one of these realisations (realisation $i$) (according to our model) is $\mathbb{P}_\beta(y = 1 | \bold{x}_i)$. By the laws of probability the probability we see all of our realisations is the product of the probabilities. Logically since these are the values that we have observed in reality, we want our model to output a probability of 1 for all of these classifications occuring i.e we want to maximise the likelyhood of these realisations being seen so as to refelcet reality and train an accurate model. The (negative, as to minimise the negative is to maximise the positive) likelyhood, $l$ is then given by:

$$
l(\beta|\bold{x}_1, \dots, \bold{x_n}) = - (\prod_{y_i = 1}^{n}\frac{1}{1+e^{-\bold{x}_i.\bold{\beta}}})(\prod_{y_i = 0}^{n}\frac{1}{1+e^{-\bold{x}_i.\bold{\beta}}})
$$

By taking the log of both sides we can turn products into sums:

$$
L(\beta|\bold{x}_1, \dots, \bold{x_n}) = log(l) = -\sum_{i = 1}^{n}y_i\log(\frac{1}{1+e^{-\bold{x}_i.\bold{\beta}}}) + (1-y_i)\log(\frac{1}{1+e^{-\bold{x}_i.\bold{\beta}}}) \\= \text{The binary cross entropy loss function!}
$$

Where the factors $y_i$ and $(1-y_i)$ are relics of the top product in $l$, as one is $0$ when the other is $1$.

We can then minimise this log likelyhood (which is a convex function and thus has a global minimum point) using gradient descent to find the optimal set of paramaters $\bold{\beta}$.

## Making predictions

Once we have found the optimal parameters, we can make predictions. The sigmoid function can be interpereted as outputting probabilities, and in this case it outputs the logistic regression model's confidence score ($1$ being certain and $0.5$ being uncertain), thus we must define a `threshold probability` for which we accept the models response, and thus classify the points. Hence we say the following, for some threshold value $t \in (0,1)$:

$$\frac{1}{1 + e^{-\bold{x}.\bold{\beta}}} \geq t$$

If this is true then we assign the variable class $1$. Otherwise we assign the variable class $0$.

### Decision boundary

The decision barrier exists within the plane of the data, it is a line that will ideally perfectly visually separate class $1$ from class $0$. To calculate the equation of the boundary given the threshold $t$ we look at the original equation using the `logit` function. Define $c = \log{\frac{t}{1-t}}$

$$
\bold{\beta}.\bold{x} = c \implies \\
x_n = \frac{c}{\beta_n} - \frac{1}{\beta_n}\sum_{i = 0 }^{n-1}\beta_ix_i
$$

