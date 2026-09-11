# Multi-Layer Perceptron (MLP) Notation & Mathematics

A reference for indexing, shapes, vectorization conventions, forward/backward propagation, and parameter calculation in Deep Feedforward Networks.

## 1. Network Topology & Indexing Conventions

Let an $L$-layer network be defined such that:

- **Layer $0$ ($l = 0$):** Input layer (carries features, contains no trainable parameters).
- **Layers $l \in \{1, 2, \dots, L-1\}$:** Hidden layers.
- **Layer $L$ ($l = L$):** Output layer.
- **$n^{[l]}$ or $d_l$:** Number of neurons in layer $l$.

## 2. Dataset Notation

| Notation                             | Dimension / Shape                      | Description                                     |
| :----------------------------------- | :------------------------------------- | :---------------------------------------------- |
| $\mathbf{X}$                         | $(m \times n)$ or $(m \times n^{[0]})$ | Full feature matrix ($m$ samples, $n$ features) |
| $\mathbf{x}^{(i)}$ or $\mathbf{X}_i$ | $(1 \times n)$ or $(n \times 1)$       | Feature vector of the $i$-th training sample    |
| $\mathbf{y}$                         | $(m \times 1)$ or $(m \times n^{[L]})$ | Ground truth label matrix/vector                |
| $m$                                  | Scalar ($\mathbb{Z}^+$)                | Total number of observations / rows in dataset  |
| $n = n^{[0]}$                        | Scalar ($\mathbb{Z}^+$)                | Total number of input features / columns        |

## 3. Layer-Wise Node & Parameter Notations

### A. Scalars (Single Neuron & Weight Level)

#### 1. Weights: $W^{[l]}_{j, k}$ vs $W^{[l]}_{i, j}$

There are two prevalent indexing conventions in literature. **Convention 1 is standard in modern deep learning frameworks (e.g., Andrew Ng / PyTorch / TensorFlow) because it avoids transposing weights during forward passes.**

- **Convention 1 (Destination-First: $\text{to } j \leftarrow \text{from } k$):**
  $$w^{[l]}_{j, k}$$
  - Superscript $[l]$: Target layer (the layer the weight connects **into**).
  - Subscript $j$: Target neuron in layer $l$.
  - Subscript $k$: Source neuron in layer $l-1$.

- **Convention 2 (Source-First: $\text{from } i \rightarrow \text{to } j$):**
  $$w^{[l]}_{i, j}$$
  - Superscript $[l]$: Target layer.
  - Subscript $i$: Source neuron in layer $l-1$.
  - Subscript $j$: Target neuron in layer $l$.

#### 2. Biases: $b^{[l]}_j$

- Superscript $[l]$: Layer index ($l \in \{1, \dots, L\}$).
- Subscript $j$: The specific neuron index in layer $l$ ($j \in \{1, \dots, n^{[l]}\}$).

#### 3. Pre-Activation Linear Sum: $z^{[l]}_j$

The net linear input entering neuron $j$ of layer $l$:
$$z^{[l]}_j = \sum_{k=1}^{n^{[l-1]}} w^{[l]}_{j, k} a^{[l-1]}_k + b^{[l]}_j$$

#### 4. Post-Activation Output: $a^{[l]}_j$ (or $O^{[l]}_j$)

The activated signal emitted by neuron $j$ of layer $l$:
$$a^{[l]}_j = g^{[l]}\left(z^{[l]}_j\right)$$
_(where $g^{[l]}(\cdot)$ represents the activation function for layer $l$, e.g., ReLU, Sigmoid, Tanh, Softmax)._

## 4. Matrix & Vector Forms

### Dimensions per Layer $l$:

- **Weight Matrix $\mathbf{W}^{[l]}$:**
  - _Under Destination-First ($j \leftarrow k$):_ shape is **$(n^{[l]} \times n^{[l-1]})$**
  - _Under Source-First ($i \rightarrow j$):_ shape is **$(n^{[l-1]} \times n^{[l]})$**
- **Bias Vector $\mathbf{b}^{[l]}$:** shape is **$(n^{[l]} \times 1)$** (or $(1 \times n^{[l]})$ when broadcasting across rows)
- **Pre-Activation Vector $\mathbf{z}^{[l]}$:** shape is **$(n^{[l]} \times 1)$**
- **Activation Vector $\mathbf{a}^{[l]}$:** shape is **$(n^{[l]} \times 1)$**
  - For input layer: $\mathbf{a}^{[0]} = \mathbf{x}$


## 5. Forward Propagation Equations

### For a Single Sample $\mathbf{x} \in \mathbb{R}^{n^{[0]} \times 1}$:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

$$\text{Final Prediction } \hat{\mathbf{y}} = \mathbf{a}^{[L]}$$

### Vectorized over Batch of $m$ Samples ($\mathbf{A}^{[0]} = \mathbf{X}^T \in \mathbb{R}^{n^{[0]} \times m}$):

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$

_(where $\mathbf{b}^{[l]}$ is broadcast column-wise across $m$ columns)._


## 6. Backward Propagation Notation (Gradients)

Using the shorthand differential operator $d\mathbf{V} = \frac{\partial \mathcal{L}}{\partial \mathbf{V}}$:

| Gradient Term         | Shape                        | Mathematical Meaning                                                                             |
| :-------------------- | :--------------------------- | :----------------------------------------------------------------------------------------------- |
| $d\mathbf{Z}^{[l]}$   | $(n^{[l]} \times m)$         | Error term / sensitivity $\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{[l]}}$ |
| $d\mathbf{W}^{[l]}$   | $(n^{[l]} \times n^{[l-1]})$ | $\frac{1}{m} d\mathbf{Z}^{[l]} (\mathbf{A}^{[l-1]})^T$                                           |
| $d\mathbf{b}^{[l]}$   | $(n^{[l]} \times 1)$         | $\frac{1}{m} \sum_{\text{cols}} d\mathbf{Z}^{[l]}$                                               |
| $d\mathbf{A}^{[l-1]}$ | $(n^{[l-1]} \times m)$       | $(\mathbf{W}^{[l]})^T d\mathbf{Z}^{[l]}$                                                         |


## 7. Trainable Parameter Calculation

For any fully connected (dense) layer $l$:
$$\text{Weights per layer } l = n^{[l-1]} \times n^{[l]}$$
$$\text{Biases per layer } l = n^{[l]}$$
$$\text{Total Parameters in layer } l = (n^{[l-1]} \times n^{[l]}) + n^{[l]} = n^{[l]} \times (n^{[l-1]} + 1)$$

$$\text{Total Network Parameters} = \sum_{l=1}^{L} \left[ n^{[l]} \cdot (n^{[l-1]} + 1) \right]$$


### Step-by-Step Example

#### Architecture:

- **Input Layer ($l=0$):** $n^{[0]} = 4$ features
- **Hidden Layer 1 ($l=1$):** $n^{[1]} = 3$ neurons
- **Hidden Layer 2 ($l=2$):** $n^{[2]} = 2$ neurons
- **Output Layer ($l=3$):** $n^{[3]} = 1$ neuron

#### Parameter Breakdown Table:

| Layer              | Input Units ($n^{[l-1]}$) | Output Units ($n^{[l]}$) | Weight Matrix Shape |   Weights Count   | Biases Count | Total Layer Parameters |
| :----------------- | :-----------------------: | :----------------------: | :-----------------: | :---------------: | :----------: | :--------------------: |
| **Hidden Layer 1** |            $4$            |           $3$            |   $(3 \times 4)$    | $4 \times 3 = 12$ |     $3$      | $12 + 3 = \mathbf{15}$ |
| **Hidden Layer 2** |            $3$            |           $2$            |   $(2 \times 3)$    | $3 \times 2 = 6$  |     $2$      |  $6 + 2 = \mathbf{8}$  |
| **Output Layer**   |            $2$            |           $1$            |   $(1 \times 2)$    | $2 \times 1 = 2$  |     $1$      |  $2 + 1 = \mathbf{3}$  |
| **Total**          |             —             |            —             |          —          |     **$20$**      |   **$6$**    |   **$\mathbf{26}$**    |
