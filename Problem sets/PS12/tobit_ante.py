import numpy as np 
from scipy.stats import norm
from numpy import linalg as la

name = 'Tobit'

def q(theta, y, x): 
    return -loglikelihood(theta,y,x) # Fill in 

def loglikelihood(theta, y, x): 
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    sig = np.abs(theta[-1]) # take abs() to ensure positivity (in case the optimizer decides to try it)
    
    xb = x@b
    pred = y-xb

    phi= norm.pdf(pred/sig) # fill in

    Phi = norm.cdf(xb/sig) # fill in
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)

    ll =  (y==0)*np.log(1-Phi)+(y>0)*np.log(1/sig*phi)  # fill in, HINT: you can get indicator functions by using (y>0) and (y==0)

    return ll



def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    N,K = x.shape
    b_ols = la.inv(x.T@x)@x.T@y
    res = y - x@b_ols # fill in
    sig2hat = (res.T @res)/(N-K) # fill in
    sighat = np.sqrt(sig2hat) # our convention is that we estimate sigma, not sigma squared
    theta0 = np.append(b_ols, sighat)
    return theta0 

def predict(theta, x): 
    '''predict(): the expected value of y given x 
    Returns E, E_pos
        E: E(y|x)
        E_pos: E(y|x, y>0) 
    '''
    # Fill in 
    b = theta[:-1]
    sig = theta[-1]

    xb = x@b
    cdf = norm.cdf(xb/sig)
    pdf = norm.pdf(xb/sig)
    IMR = cdf/pdf

    E = xb *cdf + sig*pdf
    Epos = xb + sig*IMR
    return E, Epos

def sim_data(theta, N:int): 
    b = theta[:-1]
    sig = theta[-1]
    K=b.size

    # FILL IN : x will need to contain 1s (a constant term) and randomly generated variables
    xx = np.random.normal(size=(N,K-1))
    oo = np.ones((N,1))
    x  = np.hstack([oo,xx])

    eps = np.random.normal(size=N)
    y_lat=  x @ b + eps
    #print("y_lat.shape:", y_lat.shape)
    #print("eps shape", eps.shape)
    assert y_lat.ndim==1
    y = np.fmax(y_lat,0)

    return y,x
