# Naive bayes

The Naive Bayes algoirthm is based on baysean statistics, this means that we view the data as having a prior probability distribution, and use the learnings from the data to define a posterior probability distribution. The algorithm is naive as it does not consider the order of data being entered into it, and thus is highly biased towards non ordered words.

## The maths

The naive bayes classifier relies on bayes theorem:

$$
\mathbb{P}(A|B) = \frac{\mathbb{P}(B|A)\mathbb{P}(A)}{\mathbb{P}(B)}
$$

Thus we can say that $A = y$ which is the classification variable and $B = X$ is the real data. Thus for $n$ features $X = x_1, \dots, x_n$

$$
\mathbb{P}(y|X = x_1, \dots, x_n) = \prod_{i=1}^{n}\frac{\mathbb{P}(y|X = x_i)\mathbb{P}(y)}{\mathbb{P}(X = x_i)}
$$

Thus we can say that

$$
\mathbb{P}(y|X = x_1, \dots, x_n) \propto \mathbb{P}(y)\prod_{i = 1}^{n}\mathbb{P}(X = x_i|y)
$$

So essentially we reframe the problem from being "What is the probability of seeing evidence $X$ given that the data belongs to class $y$?" to "What is the probability of seeing class $y$ given that I've seen the following evidence $X$?" by using bayes' theorem!

Thus for each datapoint, $X$ we will calculate $\mathbb{P}(y = y_i| X)$ and take the highest probability.

Note: to avoid the case in which there are numbers that were not encountered in the training set, we add some shifting factor $\alpha$ to each indepdenent variable in a tesitng set.

## Example

Suppose we have the following data example that contains data that represents wether a game was played or not depending on factors:

| Outlook  | Temperature | Humidity | Windy | Play |
|----------|-------------|----------|-------|------|
| sunny    | hot         | high     | false | NO   |
| sunny    | hot         | high     | true  | NO   |
| overcast | hot         | high     | false | YES  |
| rainy    | mild        | high     | false | YES  |
| rainy    | cool        | normal   | false | YES  |
| rainy    | cool        | normal   | true  | NO   |
| overcast | cool        | normal   | true  | YES  |
| sunny    | mild        | high     | false | NO   |
| sunny    | cool        | normal   | false | YES  |
| rainy    | mild        | normal   | false | YES  |
| sunny    | mild        | normal   | true  | YES  |
| overcast | mild        | high     | true  | YES  |
| overcast | hot         | normal   | false | YES  |
| rainy    | mild        | high     | true  | NO   |

Source: https://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn06-notes-nup.pdf


Here $X = (\text{Outlook}, \text{Temperature}, \text{Humidity}, \text{Windy})$ is our vector of independent variables and $y \in \{\text{YES}, \text{NO}\}$, we can now calculate our prior probabilities and conditional probabilites.

We define our prior probablities as follows:

$$\mathbb{P}(y = \text{NO}) = \frac{5}{14}$$
$$\mathbb{P}(y = \text{YES}) = \frac{9}{14}$$

Now we can look at the categorical value  Outlook and find it's counts and it's conditional probabilities.

| Outlook  | YES |  NO |
|----------|---------|--------|
| sunny    | 2/9     | 3/5    |
| overcast | 4/9     | 0/5    |
| rainy    | 3/9     | 2/5    |

Then given the following real data:

| Outlook  | Temperature | Humidity | Windy | Play |
|----------|-------------|----------|-------|------|
| sunny    | cool         | high     | true | ??   |

We can calculate the following:

$$
\mathbb{P}(\text{Play} = \text{YES}| X) \propto \mathbb{P}(\text{Play}=\text{YES}) \\· \mathbb{P}(\text{Outlook}= \text{sunny} | \text{Play}=\text{YES}) \\· \mathbb{P}(\text{Temperature} = \text{cool} | \text{play}=\text{YES})
\\· \mathbb{P}(\text{Humidity} = \text{high} | \text{play} = \text{YES}) \\· \mathbb{P}(\text{Windy} = \text{true} | \text{play} = \text{YES})
$$

And then likewise for when Play is equal to NO, thus we can compare both probabilities.

