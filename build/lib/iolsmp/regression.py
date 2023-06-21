import numpy as np
import statsmodels.api as sm
import pandas as pd

def Y_hat(beta, delta, X, Y):
  Y = Y.reshape(-1,1)
  return np.log(Y + delta * np.exp(np.dot(X,beta))) - c(beta, delta, X, Y)

def Y_i_hat_plmp(beta, delta,X, Y):
  Y = Y.reshape(-1,1)
  return np.log(Y + delta * np.exp(np.dot(X,beta))) - c_plmp(beta, delta, X, Y)

def c(beta, delta, X, Y):
  return np.log(delta + Y * np.exp(-np.dot(X, beta))) - 1/(1 + delta)*(Y * np.exp(-np.dot(X, beta)) - 1)

def c_plmp(beta, delta, X, Y):
  return np.log(delta + Y * np.exp(-np.dot(X,beta))) - 1/(1 + delta)*(Y - np.exp(np.dot(X,beta)))

def naive_solution(X,Y,delta):
  Y = Y.reshape(-1,1)
  Y_h = np.log(Y+delta*np.ones((Y.shape[0],1)))
  model = sm.OLS(Y_h, X)
  results = model.fit()
  return results.params.reshape(-1,1)

def linear_regression(X, Y, delta=5, eps=1e-06, method="iOLS", se="robust", cluster=None):
    try:
        # beta0 = 0.1 * np.ones((X.shape[1]+1, 1))
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        beta0 = naive_solution(X, Y, 1)
        inv_XtX = np.linalg.inv(np.dot(X.T, X))
        while True:
            if method == "iOLS":
                Y_h = Y_hat(beta0, delta, X, Y)
            elif method == "PLMP":
                Y_h = Y_i_hat_plmp(beta0, delta, X, Y)
            else:
                raise ValueError("Method should be either iOLS or PLMP: {}".format(method))
            beta1 = np.dot(inv_XtX, np.dot(X.T, Y_h))
            if np.linalg.norm(beta1 - beta0) < eps:
                break
            beta0 = beta1

        model = sm.GLM(Y_h, X)
        if se == "robust":
            model.fit_options = {"maxiter": 1}
            results = model.fit(cov_type='HC3', start_params=beta0.flatten())
            print(results.summary())
            cov = results.bse
            u = np.random.normal(0, 0.5, (Y.shape[0], 1))
            nu = 1 + u / np.exp(np.dot(X, beta0))
            W = np.diag(delta / (delta + np.squeeze(nu)))
            n = X.shape[0]
            W_util = ((1 / n) * (X.T @ (np.eye(n) - W) @ X))
            omega_h = np.linalg.inv(W_util) @ cov @ W_util
            return beta0, omega_h
        elif se == "clustered":
            if cluster is None:
                raise ValueError("Cluster variable must be provided for clustered standard errors.")
            model.fit_options = {"maxiter": 1}
            results = model.fit(cov_type='cluster', start_params=beta0.flatten(), cov_kwds={'groups': cluster})
            cluster_cov = np.diag(results.bse)
            print(results.summary())
            return beta0, cluster_cov
        else:
            raise ValueError("Invalid value for se variable: {}".format(se))
    except Exception as e:
        print("An error occurred: {}".format(e))
