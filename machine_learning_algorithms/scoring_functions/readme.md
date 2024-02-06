# Evaluation of ML models

To see how well models perform when trained on given data we need to have an idea of how to evaluate them.

## Confusion matrix

The confusion matrix is a way of measuring the number of true positives, false positives and false negatives for a set of classes.

* True positives: When the predicted class is equal to the true class
* False positives: When the predicted class is not equal to the true class because the true class was negative, but this class was positive
* False negative: When the predicted class is not equal to the true class, because the true class was positive but this class was negative

This makes sense for 2 classes, say 0 meaning negative and 1 meaning positive and will look something like this:

$$
\begin{pmatrix}
TP_0 & FN \\
FP & TP_1
\end{pmatrix}
$$

Where the columns are predictions and the rows are the truth, so when column $i = j$ we are looking at full positives, but when row $i \neq j$ we're looking at false positives or false negatives.

We can define **Accuracy** as the number of correct predictions over the total number of predictions, which is quite similar to precision.

For more than $2$ variables these terms lose their meaning. Suppose we have $n$ classes, then the confusion matrix is an $n \times n$ matrix $C$ where $C_{ii}$ is the number of correct classifications of class $i$. Then $C_{ij}$ when $i \neq j$ can be interpereted as the number of times the model predicts class $i$ when the true class is class $j$, thus we can define the true positives and false positives and false negatives for each class as follows:

$$
\text{TP}_i = C_{ii} \\
\text{FP}_i = \sum_{j \neq i}^{n}C_{ij} \\
\text{FN}_i = \sum_{j \neq i}^{n}C_{ji}
$$

Here $\text{FP}_i$ is the sum of the $i'th$ row of $C$ which is the total number of times that we predict class $i$ when the true class is not class $i$, and $\text{FN}_i$ is the sum of the $i'th$ column of $C$ which is the total number of times that the true class is class $i$ but we do not predict it.

Thus we can define **precision** and **recall** as follows:

$$
\text{precision}_{\text{class=i}}  = \frac{TP_i}{TP_i + FP_i}
$$
$$
\text{recall}_{\text{class=i}}  = \frac{TP_i}{TP_i + FN_i}
$$

Precision can be interpereted as what proportion of positive identifications was actually correct?, if the precision is $1$ then the model has correctly identified all positives. Recall can be interpereted as what proportion of actual positives was identified correctly, thus if recall is 1 then all of the positives have been identified correctly.
