# Perceptron neural network

A perceptron neural network aims to replicate how the brain works, replacing neurons with perceptrons! Spruce up.

## The maths

A perceptron can be defined as a very simple function. Suppose we have a vector of weights $\bold{w}$ and a bias $b$, and an activation function $\sigma: \mathbb{R} \rightarrow \mathbb{R}$, $p(\bold{x}) = \sigma(\bold{x}.\bold{w}+b)$. It is just a linear function that is wrapped in a (potentially) non-linear activation function.

So far we can only use one perceptron to essentially be a glorified non-linear function. So how can we allow our network to learn more in depth patterns?!

We now define the concept of a perceptron neural network, a network is comprised on $n$ layers, within the $i$'th layer there are $l_i$ perceptrons. The $1$st layer is the input layer and the $n$th layer is the output layer. Each perceptron has a bias and an output. Between the $i-1$th and $i$th layer there are $l_i$ biases and $l_{i-1} \times l_{i}$ weights, we contain the biases in a vector: $\bold{b}_i \in \mathbb{R}^{l_i}$ and the weights in a matrix: $W_{i-1,i}$ in which $(W_{i-1,i})_{k,l} = w_{k,l}$ where $w_{k,l}$ is the weight between the kth perceptron in the $i-1$th layer and the $l$th perceptron in the $i$th layer. Then we define $\Omega \in \mathbb{R}^{l_i \times l_i}$ where $\Omega = \text{diag}(\sigma)$.