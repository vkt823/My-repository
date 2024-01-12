import numpy as np
from numpy import linalg as la
from scipy.stats import norm
      
def q(beta,y,x):
    return - loglik_probit(beta,y,x)

def loglik_probit(beta, y, x):
    z =  x@beta # Fill in 
    G = norm.cdf(z) # Fill in  : remember Probit uses a normal distribution

    # Make sure that no values are below 0 or above 1.
    h = np.sqrt(np.finfo(float).eps)
    G = np.clip(G, h, 1 - h)

    # Make sure g and y is 1-D array
    G = G.reshape(-1, )
    y = y.reshape(-1, )

    ll = (y==1)*np.log(G)+(y==0)*np.log(1.0-G) # Fill In
    return ll

def starting_values(y,x):
    return la.solve(x.T @ x, x.T @ y )