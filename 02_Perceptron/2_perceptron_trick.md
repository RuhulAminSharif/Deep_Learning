# The Perceptron Trick & Decision Boundary Geometry

In 2D space, the decision boundary of a perceptron is a straight line. The **Perceptron Trick** is the geometric algorithm used to iteratively nudge this line until it correctly separates positive and negative classes.

## 1. Mathematical Notation & Equivalence

A 2D line can be written in Cartesian form or standard machine learning vector notation:

$$\underbrace{A x_1 + B x_2 + C = 0}_{\text{Algebraic Form}} \iff \underbrace{w_1 x_1 + w_2 x_2 + w_0 = 0}_{\text{ML Convention}}$$

By introducing a dummy bias input $x_0 = 1$:

$$w_0 x_0 + w_1 x_1 + w_2 x_2 = 0 \iff \sum_{i=0}^{d} w_i x_i = 0$$

In compact matrix / vector notation:

$$\mathbf{W}^T \mathbf{X} = 0$$

Where:
$$\mathbf{W} = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \end{bmatrix}, \quad \mathbf{X} = \begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 1 \\ x_1 \\ x_2 \end{bmatrix}$$

## 2. Identifying Positive and Negative Regions

Given the line $L(x_1, x_2) = w_1 x_1 + w_2 x_2 + w_0 = 0$, any point $P(p_1, p_2)$ in the space divides into one of three states:

$$L(p_1, p_2) = w_1 p_1 + w_2 p_2 + w_0$$

- **Positive Region ($+$):** $\mathbf{W}^T \mathbf{X} > 0$
- **Negative Region ($-$)**: $\mathbf{W}^T \mathbf{X} < 0$
- **On the Line:** $\mathbf{W}^T \mathbf{X} = 0$

### Test with Origin $(0, 0)$

To quickly label regions visually:

1. Substitute $(0, 0)$ into the equation: $L(0, 0) = w_0$ (or $C$).
2. If $w_0 > 0$, the half-plane containing $(0, 0)$ is the **positive region**.
3. If $w_0 < 0$, the half-plane containing $(0, 0)$ is the **negative region**.

## 3. Geometric Transformations: Role of Coefficients

Rearranging $A x + B y + C = 0$ to slope-intercept form ($y = mx + c$):

$$y = -\frac{A}{B}x - \frac{C}{B}$$

- **Slope ($m$):** $-\frac{A}{B}$ (Governed by weights $w_1, w_2$)
- **Intercept ($c$):** $-\frac{C}{B}$ (Governed by bias $w_0$ relative to $w_2$)

### Effect of Parameter Changes

| Parameter                          | Transformation Effect                                                                                                                                                                   |
| :--------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bias ($C$ or $w_0$)**            | **Translation (Shift without rotation):**<br>• Increasing $C$ shifts the line in the direction of its normal vector.<br>• Decreasing $C$ shifts the line opposite to the normal vector. |
| **Weights ($A, B$ or $w_1, w_2$)** | **Rotation (Steering the angle):**<br>• Changes the normal vector $\mathbf{n} = [A, B]^T$, rotating the line around an axis to adjust the boundary slope.                               |

## 4. The Perceptron Update Rule (The "Trick")

When a point $\mathbf{X}_i$ is misclassified, we modify $\mathbf{W}$ so that the boundary moves **toward** the misclassified point.

To prevent overshooting and ensure smooth convergence, we scale the step size using a **learning rate** ($\eta \in (0, 1]$).

### Case 1: Positive Point in Negative Region

- **Condition:** True label $y_i = +1$, but prediction $\hat{y}_i = -1$ ($\mathbf{W}^T \mathbf{X}_i < 0$).
- **Goal:** Increase $\mathbf{W}^T \mathbf{X}_i$ to make it positive.
- **Update Rule:**
  $$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} + \eta \mathbf{X}_i$$

$$\begin{bmatrix} w_0 \\ w_1 \\ w_2 \end{bmatrix}_{\text{new}} = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \end{bmatrix}_{\text{old}} + \eta \begin{bmatrix} 1 \\ x_{i1} \\ x_{i2} \end{bmatrix}$$

### Case 2: Negative Point in Positive Region

- **Condition:** True label $y_i = -1$, but prediction $\hat{y}_i = +1$ ($\mathbf{W}^T \mathbf{X}_i > 0$).
- **Goal:** Decrease $\mathbf{W}^T \mathbf{X}_i$ to make it negative.
- **Update Rule:**
  $$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \eta \mathbf{X}_i$$

$$\begin{bmatrix} w_0 \\ w_1 \\ w_2 \end{bmatrix}_{\text{new}} = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \end{bmatrix}_{\text{old}} - \eta \begin{bmatrix} 1 \\ x_{i1} \\ x_{i2} \end{bmatrix}$$

### Unified Update Formulation

Using $y_i \in \{-1, +1\}$:

$$\mathbf{W} \leftarrow \mathbf{W} + \eta (y_i - \hat{y}_i) \mathbf{X}_i$$

_(If classification is correct, $y_i - \hat{y}_i = 0$, so no update occurs)._

## 5. Pseudocode / Algorithm

```python
import numpy as np

# Hyperparameters
epochs = 1000
learning_rate = 0.01

# Initialize weights [w0, w1, w2]
W = np.zeros(3)

for epoch in range(epochs):
    # Randomly select an index i
    idx = np.random.randint(0, len(X))
    Xi = np.insert(X[idx], 0, 1)  # Xi = [1, x1, x2]
    yi = y[idx]                   # True label: +1 or -1

    # Predict: step activation on dot product
    score = np.dot(W, Xi)
    y_pred = 1 if score >= 0 else -1

    # Apply Perceptron Trick
    if yi == 1 and y_pred == -1:
        W = W + learning_rate * Xi
    elif yi == -1 and y_pred == 1:
        W = W - learning_rate * Xi
```
