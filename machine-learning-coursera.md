# Machine Learning, Coursera Course by Andrew Ng

## Introduction

### Supervised learning
**Supervised learning** is probably the most common form of machine learning. When an algorithm is provided training data with "correct" answers, it is supervised learning.

Supervised learning often solves one of two problems: regression or classification.

**Regression** models predict a continuous outcome. 

**Classification** models predict a categorical outcome.

### Unsupervised learning
**Unsupervised learning** is when the algorithm is not trained on correct answers prior to making new predictions.

Unsupervised learning often solves the problems of clustering or the cocktail party problem.

The cocktail party problem occurs when data contain mixed phenomena. The cocktail party problem is to separate out each phenomenon from the data.

## Linear Regression with One Variable
Linear regression can be approached as a machine learning regression model to predict a real-valued output.

### Model representation
Notation

- **m** = number of training examples
- **x** = feature (a.k.a. input or independent variable)
- **y** = target (a.k.a. output or dependent variable)

The generic model workflow is **training data** are input into a **learning algorithm**, which produces a function **h** that maps the features to the target. The features from the non-training data are then fed into the function **h** which maps the features to target values. These target values are the predictions.

The function **h** is referred to as the **hypothesis function** that maps the features to the targets.

A linear regression hypothesis function can be represented as 

$$h_{0}(x)=\theta_{0} + \theta_{1}(x)$$

A **central problem in machine learning** is how to represent the function **h**.

### Cost function
The cost function will help us define the best possible regression line for our data. In other words, the cost function helps us choose the best values for $\theta_{0}$ and $\theta_{1}$.

To do this, we use a minimization function as the cost function. We choose the thetas so each predicted y is as close to each actual y as possible. We minimize the sum of the squared differences between predicted y and actual y in the training data.

This particular cost function is called the squared error function.

In summary, we need four things for a linear regression model:

1. Hypothesis function
2. Parameters of the hypothesis function
3. Cost function
4. Goal

In a typical linear regression, the corresponding things are:

1. $h_{0}(x)=\theta_{0} + \theta_{1}(x)$
2. $\theta_{0}, \theta_{1}$
3. $J(\theta_{0},\theta_{1})=(\frac{1}{2m})\sum_{i = 1}^{m}(\theta_{0}(x^{(i)})-y^{(i)})^2$
4. $minimize(\theta_{0}, \theta_{1}): J(\theta_{0}, \theta_{1})$

The **hypothesis function** is a function of **x**.

The **cost function** is a function of the parameters.

### Parameter learning

#### Gradient descent
Key implementation points

- Gradient descent iterates through values of the parameters by calculating each parameter and then simultaneously updating all parameter values for the next iteration.
- The magnitude of the learning rate determines the speed and susceptibility of the descent to overshooting. (Note linear regression cost function is always concave so getting trapped in a local minimum is not a problem.)
- However, if the learning rate is too large, the algorithm might not converge even with a concave cost function.

Gradient descent is an algorithm that can be used to minimize the linear regression cost function.

It works by trying different combinations of the thetas until it reaches a minimum of the cost function. The cost function value descends along a gradient defined by different combinations of the thetas.

One weakness of this approach is if the cost function has multiple local minima. It is possible for the gradient descent to get stuck in a minimum that is not the global minimum for the entire function.

The gradient descent algorithm is repeated until it converges: $$\theta_{j}:=\theta_{j}-\alpha\frac{\partial u}{\partial u\theta_{j}}J(\theta_{0},\theta_{1})$$

In this equation, $\alpha$ is the **learning rate** that determines how quickly the algorithm descends the gradient. Intuitively, larger values of $\alpha$ mean the algorithm learns faster, but it can miss details. the learning rate multiples the derivate term.

Mathematically, what we are doing is taking the derivative of the the cost function. The slope of the tangent lineis the derivative at the point and gives us the direction to move in. We step down the cost function in the direction with the steepest descent. The size of each step is determined by the learning rate.

If the learning rate is too small, gradient descent will be very slow. If the learning rate is too large, descent can be so fast it misses a minimum and fails to converge or perhaps even diverges.

### Gradient descent in linear regression
We can apply gradient descent to minimize the linear regression cost function.

In linear regression, the cost function will always be convex, so getting trapped in local minima is not a problem.

When the gradient descent algorithm is applied to the linear regression cost function, the cost function becomes $$\theta_{0}:=\theta_{0}-\alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x_{i})-y_{i})$$ and $$\theta_{1}:=\theta_{1}-\alpha\frac{1}{m}\sum_{i = 1}^{m}((h_{\theta}(x_{i})-y_{i})x_{i})$$

### Linear algebra review

Matrix vector multiplication multiplies a matrix times a vector. The matrix must have the same number of columns as the vector has rows. The resulting product will be a vector with length equal to the number of rows of the matrix.

Matrix matrix multiplication multiplies one matrix by another. The first matrix must have the same number of columns as the second matrix has rows. The product will have dimensions equal to the number of rows of the first matrix and the number of columns of the second matrix.

Matrix multiplication properties useful for machine learning

- Matrix multiplication is not commutative. Order of matrices matters.
- Matrix multiplication is associative. The order of multiplication steps does not matter.
- The identity matrix I is a matrix of all 0s except the diagonal, which is 1s. Multiplying a matrix by an identity matrix produces the matrix as the product. A x I = I x A = A.

Special matrix operations

- Inverse
- Transpose

If A is an m x m matrix, and if it has an inverse, then $A(A^{-1})=(A^{-1})A=I$.

The transpose of a matrix is each row of a matrix turned into a column of its transpose. If A is an m x n matrix, and B is the transpose of A, then B is an n x m matrix in which $B_{ij}=A_{ji}$.


## Linear Regression with Multiple Variables

### Multivariate linear regression
Our previous hypothesis was $h_{0}(x)=\theta_{0}+\theta_{1}x$

Now, with multiple variables, the hypothesis is $$h_{0}(x) = \theta_{0}x{0} + \theta_{1}x_{1}+\theta_{2}x_{2}+ ... + \theta_{n}x_{n}$$ where $\x_{0} is a vector of 1s and $n$ is the number of variables or features.

#### Gradient descent for multiple variables
How to use gradient descent for linear regression with multiple variables?

There is now a simultaneous updating algorithm for each feature.

Repeat until convergence: $$\theta_{0}:=\theta_{0}-\alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{i})-y^{i})x_{0}^{i}$$ $$\theta_{1}:=\theta_{0}-\alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{i})-y^{i})x_{1}^{i}$$ $$...$$ $$\theta_{n}:=\theta_{0}-\alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{i})-y^{i})x_{n}^{i}$$

#### Gradient descent in practice: Feature scaling
Feature scaling ensures all the features are on a similar scale. Gradient descent converges more quickly when features are scaled.

Often we want to scale all features to be as close as possible to the range $$-1 \le x_{i} \ge 1$$.

Approaches

- Divide each feature by the maximum value of that feature
- **Mean normalization**: replace $x_{i}$ with $x_{i} - \mu_{i}$ to make features have approximately zero mean.
- **Standardization**: replace each variable with the value minus the mean divided by standard deviation or the range of the variable.

#### Gradient descent in practice: Learning rate
The learning rate in the gradient descent update rules is $\alpha$.

The job of gradient descent is to find the value of $\theta$ that minimizes the cost function $J(\theta)$.

One useful thing is to plot $J(\theta)$ against the number of iterations. If gradient descent is working properly, $J(\theta)$ should decrease as the number of iterations increases. The plot also tells you where the decline flattens out and doing more iterations has diminishing returns.

If the plot shows $J(\theta)$ is not decreasing, it's likely the learning rate is off. Usually a smaller learning rate will be better. 

However, if the learning rate is too small, gradient descent can be very slow to converge.

To choose the learning rate, start with 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, and 1, and plot the resulting change in $J(\theta)$ against interation number.

### Computing Parameters Analytically

#### Normal equation
For some linear regression, the normal equation will be the best for estimating the parameters. Rather than use gradient descent to solve for $\theta$, the normal equation solves for $\theta$ analytically.

The normal equation is $$\theta = (X^{T}X)^{-1}X^{T}y$$

The normal equation differs from gradient descent in the following ways:

1. Do not need to choose a learning rate
2. Does not need to iterate
3. Needs to compute $(X^{T}X)$
4. Slower when number of variables is large

One rule of thumb is to use gradient descent when number of features > 10,000.

#### Normal equation when the $(X^{T}X)$ matrix is non-invertible
Some matrices cannot be inverted. Such matrices are "singular" or "degenerate".

Two most common causes of non-invertibility:

1. Redudant featuers (linearly dependent features)
2. Too many features (number of observations less than number of features)

To address (2), delete some features, and/or use regularization.

### Octave tutorial

**Loops**
for i=1:10,
  v(i) = 2^i;
end;

i = 1;
while i <= 5,
  v(i) = 100;
  i = i+1;
end;

i=1;
while true,
  v(i) = 999;
  i = i+1;
  if i == 6,
    break;
  end;
end;

**Functions**
The cost function J($\theta$) can be written as an Octave function:

function J = costFunctionJ(X, y, theta)

% X is the "design matrix" containing training data.
% y is the class labels.

m = size(X,1);          % number of training examples
predictions = X*theta;  % predictions of hypothesis on all m examples
sqrErrors = (predictions-y).^2;     % squared errors

J = 1/(2*m) * sum(sqrErrors);

**Vectorization**
Coding using vector operations rather than iterations.

Example of gradient descent simultaneous updating for two features.

Non-vectorized would have a for-loop for each parameter.

A vectorized approach combines all three simultaneous update equations into a single vectorized operation.















