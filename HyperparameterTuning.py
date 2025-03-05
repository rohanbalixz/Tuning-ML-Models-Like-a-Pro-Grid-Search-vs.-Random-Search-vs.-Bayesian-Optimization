import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# -----------------------------
# Generate synthetic regression data
# -----------------------------
X, y = make_regression(n_samples=300, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_ridge(alpha):
    """Train Ridge with a given alpha and return test set MSE."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    return mse

# -----------------------------
# Setup search simulation parameters
# -----------------------------
n_iter = 20  # number of iterations (candidate evaluations)

# -- Grid Search --
grid_alphas = np.linspace(0.1, 100, n_iter)
grid_results = [evaluate_ridge(alpha) for alpha in grid_alphas]

# -- Random Search --
np.random.seed(0)  # for reproducibility
random_alphas = np.random.uniform(0.1, 100, n_iter)
random_results = [evaluate_ridge(alpha) for alpha in random_alphas]

# -- Bayesian Optimization (Simulated) --
# Here we start with a random candidate and then at each step we perturb the best-so-far candidate.
bayes_alphas = []
bayes_results = []
# initialize with a random candidate
best_bayes_alpha = np.random.uniform(0.1, 100)
best_bayes_mse = evaluate_ridge(best_bayes_alpha)
bayes_alphas.append(best_bayes_alpha)
bayes_results.append(best_bayes_mse)

for i in range(1, n_iter):
    # perturb the current best candidate by a normal noise (std=5)
    candidate = best_bayes_alpha + np.random.normal(0, 5)
    # clip candidate to allowed range
    candidate = np.clip(candidate, 0.1, 100)
    mse_candidate = evaluate_ridge(candidate)
    bayes_alphas.append(candidate)
    bayes_results.append(mse_candidate)
    # update best candidate if improved
    if mse_candidate < best_bayes_mse:
        best_bayes_alpha = candidate
        best_bayes_mse = mse_candidate

# -----------------------------
# Prepare animation: we’ll update plots as candidates are “evaluated”
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
methods = ['Grid Search', 'Random Search', 'Bayesian Optimization']
for ax, title in zip(axs, methods):
    ax.set_xlabel("Alpha")
    ax.set_ylabel("MSE")
    ax.set_xlim(0, 100)

# Determine overall MSE limits
all_mse = grid_results + random_results + bayes_results
mse_min, mse_max = min(all_mse), max(all_mse)
for ax in axs:
    ax.set_ylim(mse_min - 5, mse_max + 5)

# Initialize scatter plots and "best-so-far" markers for each method.
scatter_grid = axs[0].scatter([], [], color='red')
scatter_random = axs[1].scatter([], [], color='blue')
scatter_bayes = axs[2].scatter([], [], color='green')
best_point_grid, = axs[0].plot([], [], 'ko', markersize=10, label='Best so far')
best_point_random, = axs[1].plot([], [], 'ko', markersize=10, label='Best so far')
best_point_bayes, = axs[2].plot([], [], 'ko', markersize=10, label='Best so far')

# Lists to store the candidates and scores as they are evaluated.
grid_x, grid_y = [], []
random_x, random_y = [], []
bayes_x, bayes_y = [], []

# Variables to store best result so far (initialize with a high MSE)
best_grid_alpha, best_grid_mse = None, np.inf
best_random_alpha, best_random_mse = None, np.inf
# For Bayesian we already computed best_bayes_alpha, best_bayes_mse

def animate(i):
    global best_grid_alpha, best_grid_mse, best_random_alpha, best_random_mse
    if i < n_iter:
        # --- Grid Search ---
        alpha_g = grid_alphas[i]
        mse_g = grid_results[i]
        grid_x.append(alpha_g)
        grid_y.append(mse_g)
        if mse_g < best_grid_mse:
            best_grid_mse = mse_g
            best_grid_alpha = alpha_g

        # --- Random Search ---
        alpha_r = random_alphas[i]
        mse_r = random_results[i]
        random_x.append(alpha_r)
        random_y.append(mse_r)
        if mse_r < best_random_mse:
            best_random_mse = mse_r
            best_random_alpha = alpha_r

        # --- Bayesian Optimization ---
        alpha_b = bayes_alphas[i]
        mse_b = bayes_results[i]
        bayes_x.append(alpha_b)
        bayes_y.append(mse_b)
    
    # Update scatter plots with all evaluated points so far.
    scatter_grid.set_offsets(np.column_stack((grid_x, grid_y)))
    scatter_random.set_offsets(np.column_stack((random_x, random_y)))
    scatter_bayes.set_offsets(np.column_stack((bayes_x, bayes_y)))

    # Update best-so-far markers
    if best_grid_alpha is not None:
        best_point_grid.set_data([best_grid_alpha], [best_grid_mse])
    if best_random_alpha is not None:
        best_point_random.set_data([best_random_alpha], [best_random_mse])
    # For Bayesian, best is the minimum of bayes_results so far:
    if bayes_y:
        idx = np.argmin(bayes_y)
        best_bayes_alpha_current = bayes_x[idx]
        best_bayes_mse_current = bayes_y[idx]
        best_point_bayes.set_data([best_bayes_alpha_current], [best_bayes_mse_current])
    
    # Update subplot titles with iteration info.
    axs[0].set_title(f"Grid Search (Iteration {i+1}/{n_iter})")
    axs[1].set_title(f"Random Search (Iteration {i+1}/{n_iter})")
    axs[2].set_title(f"Bayesian Optimization (Iteration {i+1}/{n_iter})")
    
    return (scatter_grid, scatter_random, scatter_bayes,
            best_point_grid, best_point_random, best_point_bayes)

ani = animation.FuncAnimation(fig, animate, frames=n_iter, interval=600, blit=False)

plt.tight_layout()
plt.show()
