# Perceptron Loss Functions & Optimization

## Limitations of the Perceptron Trick

The heuristic Perceptron Trick has critical shortcomings in practice:

- **Order Sensitivity & Cycling:** If points are sampled with replacement or ordered poorly, the model may repeatedly update on the same few conflicting points, causing oscillation.
- **Suboptimal Decision Boundary:** The algorithm halts as soon as _any_ separating hyperplane is found. It does not search for the **maximum margin** (the "best" line), often yielding a boundary that passes dangerously close to training points.
- **Non-Convergent on Non-Separable Data:** If data is not strictly linearly separable, the algorithm loops indefinitely without an objective criterion for when it is "close enough."
- **No Global Objective:** It lacks a continuous mathematical function to quantify overall performance.

## Defining a Formal Loss Function

To address these limitations, we define a continuous scalar objective function $\mathcal{L}(\mathbf{w})$ that quantifies total prediction error across all $n$ samples:

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \mathcal{L}(\mathbf{w})$$

### Why Not Just Count Misclassifications (0-1 Loss)?

$$\mathcal{L}_{0-1}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{I}(y_i \neq \hat{y}_i)$$

- **Problem:** The 0-1 step loss is piecewise constant (gradient is $0$ almost everywhere) and non-differentiable at the boundary. Gradient-based optimization cannot determine which direction to nudge the boundary.

## The Perceptron Loss Function

To make the loss function differentiable with respect to $\mathbf{w}$, we measure the **signed functional distance** of misclassified points from the decision boundary.

Let the linear score be:
$$f(\mathbf{x}_i) = \mathbf{w}^T \mathbf{x}_i = w_0 x_{i0} + w_1 x_{i1} + \dots + w_d x_{id}$$

Using label convention $y_i \in \{-1, +1\}$:

- **Correct classification:** $y_i \cdot f(\mathbf{x}_i) > 0$
- **Misclassification:** $y_i \cdot f(\mathbf{x}_i) < 0$

### Mathematical Formulation

The loss on a single point is:
$$\ell(y_i, f(\mathbf{x}_i)) = \max\left(0, -y_i f(\mathbf{x}_i)\right) = \max\left(0, -y_i (\mathbf{w}^T \mathbf{x}_i)\right)$$

Averaged over $n$ training points:
$$\mathcal{L}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} \max\left(0, -y_i (\mathbf{w}^T \mathbf{x}_i)\right)$$

## Geometric Intuition

The perpendicular geometric distance from a point $\mathbf{x}_i$ to the hyperplane $\mathbf{w}^T \mathbf{x} = 0$ is:

$$d_i = \frac{|\mathbf{w}^T \mathbf{x}_i|}{\|\mathbf{w}\|}$$

- When a point is correctly classified, its penalty is $0$.
- When a point is misclassified, its penalty $-y_i (\mathbf{w}^T \mathbf{x}_i)$ is directly proportional to its distance from the boundary.
- **Geometric Objective:** Minimize the cumulative distance by which misclassified points violate the decision boundary.

## Optimization via Gradient Descent

To minimize $\mathcal{L}(\mathbf{w})$, compute the partial derivative with respect to $\mathbf{w}$:

$$
\frac{\partial}{\partial \mathbf{w}} \ell(y_i, f(\mathbf{x}_i)) =
\begin{cases}
\mathbf{0}, & \text{if } y_i (\mathbf{w}^T \mathbf{x}_i) \ge 0 \quad (\text{correctly classified}) \\
-y_i \mathbf{x}_i, & \text{if } y_i (\mathbf{w}^T \mathbf{x}_i) < 0 \quad (\text{misclassified})
\end{cases}
$$

### Weight Update Rule (Stochastic Gradient Descent):

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w})$$

For a misclassified point:
$$\mathbf{w} \leftarrow \mathbf{w} + \eta \, y_i \mathbf{x}_i$$

_This analytically justifies the heuristic Perceptron Trick as a first-order optimization method on the Perceptron loss function._

## Hinge Loss (Soft-Margin SVM)

The Perceptron loss stops penalizing points the moment they cross the boundary ($y_i f(\mathbf{x}_i) > 0$). **Hinge Loss** adds a safety buffer (margin) of $1$:

$$\mathcal{L}_{\text{Hinge}} = \max\left(0, 1 - y_i f(\mathbf{x}_i)\right)$$

- Points must not only be on the correct side, but at least a distance of $1$ away from the boundary to incur zero loss.
- Forms the objective function of **Support Vector Machines (SVM)**.

## The Sigmoid Function & Binary Cross-Entropy

Instead of outputting hard threshold values $\{-1, +1\}$, the perceptron score $z = \mathbf{w}^T \mathbf{x}$ can be mapped into a calibrated probability $\hat{y} \in (0, 1)$ using the **Sigmoid (logistic) function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

With continuous probabilities, we optimize using **Binary Cross-Entropy (Log Loss)** with labels $y_i \in \{0, 1\}$:

$$\mathcal{L}_{\text{BCE}}(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

- Unlike the Perceptron loss, BCE penalizes confidently wrong predictions exponentially and remains strictly convex and smooth everywhere.

## Summary: Activation & Loss Combinations

| Model                   | Output Activation                      | Loss Function                     | Target Problem                      |
| :---------------------- | :------------------------------------- | :-------------------------------- | :---------------------------------- |
| **Linear Regression**   | Linear / Identity ($f(z) = z$)         | Mean Squared Error ($\text{MSE}$) | Continuous Regression               |
| **Perceptron**          | Step Function ($\text{sgn}(z)$)        | Perceptron Loss / Hinge Loss      | Binary Classification               |
| **Logistic Regression** | Sigmoid ($\sigma(z)$)                  | Binary Cross-Entropy (Log Loss)   | Binary Probabilistic Classification |
| **Softmax Regression**  | Softmax ($\text{Softmax}(\mathbf{z})$) | Categorical Cross-Entropy         | Multiclass Classification           |
