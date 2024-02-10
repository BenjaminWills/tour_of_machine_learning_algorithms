# Gaussian naive bayes

Please read the classic naive bayes readme before looking into this algorithm! It explains what conditional probabilities are and how we can use them and the fundamental assumption of independence that makes bayes naive.

Within gaussian naive bayes we assume that the conditional distribution of the independent variables $X$ given that we know the class $y = c$ is a normal distribution:

$$
\mathbb{P}(X|y = c) = \frac{1}{\sigma \sqrt{2\pi}} e^{\frac{-(X-\mu_c)^2}{2\sigma_c^2}}
$$

Where $\mu_c$ is the mean value of the $X$ values that have clas C and $\sigma_c$ is the standard deviation of the $X$ values.

Explictly if we have $n$ datapoints, and are looking at the $i$'th independent variable and the $c$'th class, suppose that $C_{i,c}$ is the number of dependent variables with calss $c$, here $\mu_{i,c},\sigma_{i,c}$ are the mean and standard deviation of variables $i$ that belong to class $c$.

$$
\mu_{i,c} = \frac{1}{C_{i,c}}\sum_{x \in c}{x} \\
\sigma_{i,c}^2 = \frac{1}{C_{i,c}}\sum_{x \in c}{(x-\mu_{i,c})^2}
$$

Thus now we can calculate our posterior distribution using these calculated quantities!