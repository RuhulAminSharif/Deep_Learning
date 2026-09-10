import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from sklearn.datasets import make_classification

save_path = "./02_Perceptron/trick"

# 1. Generate Synthetic Data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=41,
    hypercube=False,
    class_sep=10,
)

# Save initial scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap="winter", s=100)
plt.title("Initial Data Distribution")
plt.savefig(f"{save_path}/initial_scatter.png", dpi=300, bbox_inches="tight")
plt.close()


# 2. Step Activation Function
def step(x):
    return 1 if x > 0 else 0


# 3. Standard Perceptron Training
def perceptron(X, y):
    X = np.insert(arr=X, obj=0, values=1, axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1

    for i in range(1000):
        j = np.random.randint(low=0, high=100)
        y_hat = step(np.dot(X[j], weights))
        weights = weights + lr * (y[j] - y_hat) * X[j]

    return weights[0], weights[1:]


intercept_, coef_ = perceptron(X=X, y=y)

# Calculate final boundary line parameters
m_final = -(coef_[0] / coef_[1])
b_final = -(intercept_ / coef_[1])
x_input = np.linspace(-3, 3, 100)
y_input = m_final * x_input + b_final

# Save Final Static Decision Boundary
plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color="red", linewidth=3, label="Final Boundary")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="winter", s=100)
plt.ylim(-3, 2)
plt.title("Final Trained Perceptron Boundary")
plt.legend()
plt.savefig(f"{save_path}/final_boundary.png", dpi=300, bbox_inches="tight")
plt.close()


# 4. Perceptron Animation Tracking
def perceptron_animation(X, y):
    m = []
    b = []

    X = np.insert(arr=X, obj=0, values=1, axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1

    for i in range(1000):
        j = np.random.randint(low=0, high=100)
        y_hat = step(np.dot(X[j], weights))
        weights = weights + lr * (y[j] - y_hat) * X[j]

        # Prevent division by zero if weight becomes exactly 0
        w2 = weights[2] if weights[2] != 0 else 1e-5

        m.append(-(weights[1] / w2))
        b.append(-(weights[0] / w2))
    return m, b


m, b = perceptron_animation(X, y)

# Setup Animation Canvas (Fixed typo: subplots)
fig, ax = plt.subplots(figsize=(9, 5))
x_i = np.arange(-3, 3, 0.1)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="winter", s=100)

# Fixed unpacking error (line, instead of line, _)
(line,) = ax.plot(x_i, x_i * m[0] + b[0], "r-", linewidth=2)
ax.set_ylim(-3, 3)
ax.set_title("Perceptron Learning Animation")


def update(i):
    label = f"Iteration {i + 1}"
    line.set_ydata(x_i * m[i] + b[i])
    ax.set_xlabel(label)
    return (line,)


# Create animation sequence
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# 5. Save the Animation to File
# To save as MP4 video (Requires ffmpeg installed on your system):
try:
    anim.save(f"{save_path}/perceptron_training.mp4", writer="ffmpeg", fps=20)
    print("Animation successfully saved as 'perceptron_training.mp4'!")
except Exception:
    anim.save(f"{save_path}/perceptron_training.gif", writer="pillow", fps=20)
    print("Animation successfully saved as 'perceptron_training.gif'!")

plt.close()
