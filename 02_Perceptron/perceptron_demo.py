import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

save_path = "./02_Perceptron"
# 1. Load and explore data
df = pd.read_csv("./02_Perceptron/placement.csv")
print("Data Shape:", df.shape)

sns.scatterplot(x=df["cgpa"], y=df["resume_score"], hue=df["placed"])
plt.title("Placement Scatter Plot")
plt.savefig(f"{save_path}/scatterplot.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Extract features and target as NumPy arrays
X = df.iloc[:, 0:2].to_numpy(dtype=float)
y = df.iloc[:, -1].to_numpy(dtype=int)

# 3. Initialize and train the Perceptron
model = Perceptron()
model.fit(X=X, y=y)

# 4. Print weights and bias
print("Weights (W):", model.coef_)
print("Bias (b):", model.intercept_)

# 5. Plot decision boundary
plot_decision_regions(X=X, y=y, clf=model, legend=2)
plt.xlabel("CGPA")
plt.ylabel("Resume Score")
plt.title("Perceptron Decision Boundary")
plt.savefig(f"{save_path}/decision_boundary.png", dpi=300, bbox_inches="tight")  # Saves the image
plt.close()
