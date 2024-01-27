# Loss functions

Every supervised machine learning algorithm requires some sort of cost function to "learn" via optimising.

## Cross entropy

Cross entropy is incredibly useful and commonly used when assessing the performance of classification problems.

Consider the following set of data:

| height | weight | sex    | probability predicition |
|--------|--------|--------|-------------------------|
| 180    | 90     | male   | 0.4                     |
| 150    | 64     | female | 0.9                     |

What this table has are heights and weights, and then the classification of the sex of the subject. A model has predicted a 40% chance of the first subject being a male and a 90% chance of the subject being a female. 

We know that 40% is very uncertain of the model, it is almost at random chance - thus we need a way to penalise the model for high uncertainty. A suitable function for this is the $-\ln$ function, which is simply $\log$ base $e$, as $-ln(x)$ approaches $\infty$ as $x$ approaches $0$, i.e as we become less certain the value of this function gets larger and thus we are penalised more. In this example our loss for the first row would be $-ln(0.4) = 0.9163$ and for the second our loss is $-ln(0.9) = 0.1054$ which is what we'd expect as we are correctly much more certain that subject 2 is a female.

Thus a suitable formulation for a general classification problem with $n$ classes and $m$ datapoints, the probability of the observation $p(y)$ and the predicted probability $\hat{p}(\hat{y})$ is:

$$\text{Cross entropy} = -\frac{1}{m}\sum_{j = 1}^{m}\sum_{i=1}^{n}p(y_{i,j})\ln{(\hat{p}(\hat{y}_{i,j}))}$$

Note that $p(y_{i,j}) = \delta_{i,j}$ where j is the actual observed class, so in the case of row one above $p(y_1) = 1$ i.e the probability of being a male. Furhter $j$ here represents the $j$th sample from the set and $i$ represents the category output.

Imagine that we had a ground truth $[0,1,0,0]$ i.e the item is category $1$ (the index where $1$ is on the list) and a prediction of probabilities which is $[0.2,0.5,0.2,0.1]$ then the cross entropy is equal to $0*\ln(0.2)-1*\ln(0.5) - 0*\ln(0.2) - 0*\ln(0.1) = \ln(2)$.

i.e if we're confident about the correct class then $\ln{(\hat{p}(\hat{y}))}$ is generally small and we have small loss and visa versa. In the case of $m = 2$ classes then $p(y_1) = 1 - p(y_2)$ thus the sum reduces to:

$$\text{Binary cross entropy} = -\frac{1}{n}\sum_{i=1}^{n}p(y_i)\ln(\hat{p}(\hat{y}_i)) + (1-p(y_i))\ln(1-\hat{p}(\hat{y}_i))$$