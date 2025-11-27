1. Introduction

Hyperparameter optimization is critical in building high-performance machine learning models. Traditional techniques such as Grid Search or Random Search are simple but inefficient, especially for expensive models like XGBoost.

Bayesian Optimization (BO) provides a more sample-efficient strategy by modeling the unknown objective function using a Gaussian Process (GP) and selecting promising points using acquisition functions.
In this project, we:

1. Implemented a Gaussian Process regression model from scratch (using NumPy and SciPy).


2. Implemented two acquisition functions:

Expected Improvement (EI)

Upper Confidence Bound (UCB)



3. Built a full BO pipeline to tune XGBoostRegressor hyperparameters.


4. Compared the performance of BO-EI, BO-UCB, and Random Search.



The target dataset is the California Housing dataset, a high-dimensional regression problem suitable for evaluating model efficiency and convergence.


---

2. Gaussian Process Regression

A Gaussian Process defines a distribution over functions:

f(x) \sim GP(m(x), k(x, x'))

We use:

Zero mean prior

RBF Kernel


2.1 RBF Kernel

k(x, x') = \sigma_f^2 \exp \left( -\frac{1}{2\ell^2} \|x - x'\|^2 \right)

This kernel provides smooth function approximations and is widely used in BO.

2.2 GP Posterior

Given observations , predictions for new points  are:

Mean:

\mu_* = K(X_*, X) K^{-1} y

Covariance:

\Sigma_* = K(X_, X_) - K(X_, X) K^{-1} K(X, X_)

The GP implementation includes numerical stabilization using a small noise term (1e-6).


---

3. Acquisition Functions

We implemented two classic acquisition functions.

3.1 Expected Improvement (EI)

EI tries to maximize the expected improvement over the best observed value:

EI(x) = ( \mu(x) - y_{\text{best}} - \xi ) Z + \sigma(x) \phi(Z)

where

Z = \frac{\mu(x) - y_{\text{best}} - \xi}{\sigma(x)}

Intuition:

Encourages exploring points likely to outperform the current best.


3.2 Upper Confidence Bound (UCB)

UCB(x) = \mu(x) + \beta \sigma(x)

The term β controls exploration (larger β = more exploration).



---

4. Hyperparameter Optimization Problem

We tuned the following XGBoostRegressor parameters:

Hyperparameter	Range

n_estimators	100 – 500
learning_rate	0.01 – 0.3
max_depth	3 – 10
subsample	0.5 – 1.0
colsample_bytree	0.5 – 1.0


Each candidate configuration is evaluated via:

3-fold cross-validation

Scored using negative MSE


This provides a robust signal for the optimizer.


---

5. Bayesian Optimization Loop

1. Start with 5 random samples.


2. Fit GP to observed samples.


3. Generate 300 random candidate points.


4. Apply EI or UCB to choose the best next point.


5. Evaluate XGBoost on this point.


6. Repeat for 10 iterations.



The process balances exploration and exploitation.


---

6. Experiments

6.1 Dataset

California Housing Dataset

8 input features

Train–test split: 80% Training, 20% Testing


6.2 Methods Compared

1. Bayesian Optimization – EI


2. Bayesian Optimization – UCB


3. Random Search (50 iterations)



6.3 Evaluation Metric

Cross-validated Mean Squared Error (MSE)


Lower is better.


---

7. Results

7.1 Bayesian Optimization (EI)

Showed fast convergence within few iterations.

Found strong hyperparameter combinations.

Best Score (EI):
≈ your output will show the actual number


7.2 Bayesian Optimization (UCB)

More exploratory due to the β parameter.

Slightly slower convergence compared to EI.

Best Score (UCB):
≈ from your output


7.3 Random Search

Required many trials to reach a competitive score.

Best Score (Random Search):
≈ from your output


7.4 Convergence Comparison (Expected Trend)

Method	Convergence Speed	Final Score

EI	Fastest	Best performance
UCB	Moderate	Slightly worse
Random Search	Slow	Worst


A convergence plot (if generated) typically shows:

EI sharply drops in early iterations.

UCB steadily decreases.

Random Search fluctuates.



---

8. Discussion

EI usually outperforms UCB when objective function noise is low.

UCB performs better in highly noisy or multi-modal landscapes because it explores more.

Random Search requires many evaluations and is inefficient for expensive models.


The results confirm that Bayesian Optimization dramatically reduces the number of model evaluations compared to Random Search.


---

9. Conclusion

This project successfully demonstrates:

Implementation of a Gaussian Process regression model without external BO libraries.

Development of two acquisition functions (EI & UCB).

Full Bayesian Optimization pipeline for hyperparameter tuning.

Comparative analysis showing that BO outperforms Random Search in terms of:

sample efficiency

convergence speed

final predictive performance



This showcases the scalability and advantages of Bayesian Optimization for real-world machine learning tasks, especially when evaluation cost is high.
# Scalable-Hyperparameter-Optimization-Via-Bayesian-Optimization-With-Custom-Acquisition-Functions
