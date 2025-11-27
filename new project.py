import numpy as np
from scipy.stats import norm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings

warnings.filterwarnings("ignore")
class BayesianOptimizer:
    def __init__(self, bounds, acquisition="ei", random_state=42):
        self.bounds = np.array(bounds, dtype=float)
        self.acquisition = acquisition.lower()
        self.random_state = random_state
        np.random.seed(random_state)

        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

        self.X_samples = []
        self.y_samples = []

    # Evaluate XGBoost params
    def evaluate(self, params, X, y):
        model = XGBRegressor(
            n_estimators=int(params[0]),
            learning_rate=float(params[1]),
            max_depth=int(params[2]),
            subsample=float(params[3]),
            colsample_bytree=float(params[4]),
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
        score = -np.mean(
            cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=3)
        )
        return score

    # EI function (safe for minimization)
    def expected_improvement(self, mu, sigma, best):
        with np.errstate(divide='ignore'):
            improvement = best - mu
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0] = 0
        return ei

    # UCB (higher is better)
    def ucb(self, mu, sigma, kappa=2.5):
        return -mu + kappa * sigma  # since we're minimizing objective

    # Propose new point
    def propose_location(self, n_candidates=2000):
        candidates = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(n_candidates, len(self.bounds))
        )

        mu, sigma = self.gp.predict(candidates, return_std=True)
        best = np.min(self.y_samples)
        if self.acquisition == "ei":
            acq = self.expected_improvement(mu, sigma, best)
        else:
            acq = self.ucb(mu, sigma)

        return candidates[np.argmax(acq)]

    # Main optimization loop
    def run(self, X, y, init_points=5, iterations=10):
        # Initial random points
        for _ in range(init_points):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            self.X_samples.append(x0)
            self.y_samples.append(self.evaluate(x0, X, y))

        self.X_samples = np.array(self.X_samples)
        self.y_samples = np.array(self.y_samples)

        for t in range(iterations):
            print(f"Iteration {t+1}/{iterations}  |  Best so far: {np.min(self.y_samples):.4f}")

            self.gp.fit(self.X_samples, self.y_samples)
            x_next = self.propose_location()
            y_next = self.evaluate(x_next, X, y)

            self.X_samples = np.vstack([self.X_samples, x_next])
            self.y_samples = np.append(self.y_samples, y_next)
        
        best_idx = np.argmin(self.y_samples)
        return self.X_samples[best_idx], self.y_samples[best_idx]

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

bounds = [
    (100, 500),      # n_estimators
    (0.01, 0.3),     # learning_rate
    (3, 10),         # max_depth
    (0.5, 1.0),      # subsample
    (0.5, 1.0)       # colsample_bytree
]

print("\n=== Bayesian Optimization (EI) ===")
bo_ei = BayesianOptimizer(bounds, acquisition="ei", random_state=0)
best_params_ei, best_score_ei = bo_ei.run(X_train, y_train, iterations=10)
print("Best Params (EI):", best_params_ei)
print("Best Score (EI):", best_score_ei)

print("\n=== Bayesian Optimization (UCB) ===")
bo_ucb = BayesianOptimizer(bounds, acquisition="ucb", random_state=1)
best_params_ucb, best_score_ucb = bo_ucb.run(X_train, y_train, iterations=10)
print("Best Params (UCB):", best_params_ucb)
print("Best Score (UCB):", best_score_ucb)
print("\n=== Random Search Baseline ===")
best_random = float("inf")
best_param = None

for _ in range(50):
    params = [
        np.random.uniform(100, 500),
        np.random.uniform(0.01, 0.3),
        np.random.uniform(3, 10),
        np.random.uniform(0.5, 1.0),
        np.random.uniform(0.5, 1.0)
    ]

    model = XGBRegressor(
        n_estimators=int(params[0]),
        learning_rate=params[1],
        max_depth=int(params[2]),
        subsample=params[3],
        colsample_bytree=params[4],
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    score = -np.mean(
        cross_val_score(model, X_train, y_train,
                        scoring="neg_mean_squared_error", cv=3)
    )

    if score < best_random:
        best_random = score
        best_param = params
print("Best Random Params:", best_param)
print("Best Random Score:", best_random)

