import numpy as np
from numpy import linalg as la
from tabulate import tabulate
from scipy.stats import chi2

def estimate( 
        y: np.ndarray, x: np.ndarray, transform='', T:int=None
    ) -> list:
    """Uses the provided estimator (mostly OLS for now, and therefore we do 
    not need to provide the estimator) to perform a regression of y on x, 
    and provides all other necessary statistics such as standard errors, 
    t-values etc.  

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)
        >> transform (str, optional): Defaults to ''. If the data is 
        transformed in any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation.
        >>t (int, optional): If panel data, t is the number of time periods in
        the panel, and is used for estimating the variance. Defaults to None.

    Returns:
        list: Returns a dictionary with the following variables:
        'b_hat', 'se', 'se_robust', 'sigma2', 't_values', 't_values_robust, 'R2', 'cov', 'cov_robust'
    """
    
    b_hat = est_ols(y, x)  # Estimated coefficients
    residual = y - x@b_hat  # Calculated residuals
    SSR = residual.T@residual  # Sum of squared residuals
    SST = (y - np.mean(y)).T@(y - np.mean(y))  # Total sum of squares
    R2 = 1 - SSR/SST

    sigma2, cov, se = variance(transform, SSR, x, T)
    t_values = b_hat/se

    cov_robust, se_robust = variance_robust(residual, x, T)
    t_values_robust = b_hat/se_robust

    names = ['b_hat', 'se', 'se_robust', 'sigma2', 't_values', 't_values_robust', 'R2', 'cov', 'cov_robust']
    results = [b_hat, se, se_robust, sigma2, t_values, t_values_robust, R2, cov, cov_robust]
    return dict(zip(names, results))

    
def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x)@(x.T@y)

def variance( 
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        T: int
    ) -> tuple:
    """Calculates the covariance and standard errors from the OLS
    estimation.

    Args:
        >> transform (str): Defaults to ''. If the data is transformed in 
        any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation
        >> SSR (float): Sum of squared residuals
        >> x (np.ndarray): Dependent variables from regression
        >> t (int): The number of time periods in x.

    Raises:
        Exception: If invalid transformation is provided, returns
        an error.

    Returns:
        tuple: Returns the error variance (mean square error), 
        covariance matrix and standard errors.
    """

    # Store n and k, used for DF adjustments.
    K = x.shape[1]
    if transform.lower() == 'be':
        N = x.shape[0]
    else:
        N = x.shape[0]/T

    # Calculate sigma2
    if transform.lower() in ('', 'fd', 're'):
        sigma2 = (np.array(SSR/(N * T - K)))
    elif transform.lower() == 'fe':
        sigma2 = np.array(SSR/(N * (T - 1) - K))
    elif transform.lower() == 'be':
        sigma2 = np.array(SSR/(N - K))
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)

    
    return sigma2, cov, se


import numpy as np
import numpy.linalg as la

def variance_robust(
        resid: np.array,
        x: np.ndarray,
        T: int,
    ) -> tuple:
    """
    Calculates the variance-covariance matrix and standard errors of the OLS estimator
    using the robust method.

    Parameters:
        >> SSR (float): Sum of squared residuals
        >> x (np.ndarray): Dependent variables from regression

    Returns:
        tuple: Returns the robust covariance matrix and robust standard errors.
    """
    N = int(x.shape[0]/T)
    K = x.shape[1]
    Z = np.empty((N, K, K))
    for i in range(N): 
        ii_A = slice(i*T, (i+1)*T)
        Z[i, :] = x[ii_A, :].T @ resid[ii_A, :] @ resid[ii_A, :].T @ x[ii_A, :]

    bread = np.empty((K,K))

    for i in range(K):
        for j in range(K):
            bread[i, j] = np.sum(Z[:, i, j])


    omega = resid**2
    #white_rob = la.inv(x.T@x)@x.T@(omega*x@la.inv(x.T@x))
    #cov_robust = la.inv(x.T@x)@(SSR*x.T@x)@la.inv(x.T@x)
    #se_robust = np.sqrt(white_rob.diagonal()).reshape(-1, 1)

    cov_robust = la.inv(x.T@x)@bread@la.inv(x.T@x)
    se_robust = np.sqrt(cov_robust.diagonal()).reshape(-1, 1)

    return cov_robust, se_robust

def wald_test(beta: np.array,
              R: np.array,
              r: np.array,
              vcov: np.ndarray,
              ) -> tuple:
        
    
     Wald = (R@beta-r).T@(la.inv(R@vcov@R.T))@(R@beta-r)
     Q = r.shape[0]
     p_val = 1-chi2.cdf(Wald, Q)
     
     return p_val, Wald


def wald_test2(y, X, trans='', T=None, significance_level=0.05):
    """
    Perform a Wald test to assess the significance of coefficients in a regression model.

    Args:
        y (np.ndarray): Dependent variable.
        X (np.ndarray): Independent variable(s).
        trans (str, optional): Data transformation type.
        T (int, optional): Number of time periods in panel data.
        significance_level (float, optional): Significance level (default: 0.05).

    Returns:
        tuple: Wald statistic, p-value, and test result.
    """
    results_dict = estimate(y, X, transform = trans, T=T)
    b_hat = results_dict['b_hat']
    cov_robust = results_dict["cov_robust"]

    if trans.lower() in ('', 're', 'be'):
        linear_combination = np.array([0, 1, 1])
    elif trans.lower() in ('fd', 'fe'):
        linear_combination = np.array([1, 1])

    R = linear_combination.reshape(1, -1) # matrix specifying the constraint: beta1 + beta2
    r = np.array([1])       # beta1 + beta2 = 1
    

    wald_statistic = ((R @ b_hat - r).T @ np.linalg.inv(R @ cov_robust @ R.T) @ (R @ b_hat - r))[0, 0]

    # calculate critical value
    degrees_of_freedom = R.shape[0]
    critical_value = chi2.ppf(1 - significance_level, degrees_of_freedom)
    
    # calculate p-value
    p_value = 1 - chi2.cdf(wald_statistic, degrees_of_freedom)
    
    if wald_statistic > critical_value:
        result = "Reject the null hypothesis"
    else:
        result = "Fail to reject the null hypothesis"
    
    return wald_statistic, p_value, result



def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values", "Se (robust)", "t-values (robust)"],
        title="Results",
        _lambda:float=None,
        **kwargs
    ) -> None:
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and t-values. The number of coefficients must be the
    same length as the labels.

    Args:
        >> labels (tuple): Touple with first a label for y, and then a list of 
        labels for x.
        >> results (dict): The results from a regression. Needs to be in a 
        dictionary with at least the following keys:
            'b_hat', 'se', 't_values', 'R2', 'sigma2'
        >> headers (list, optional): Column headers. Defaults to 
        ["", "Beta", "Se", "t-values", "Se (robust)", "t-values (robust)"].
        >> title (str, optional): Table title. Defaults to "Results".
        _lambda (float, optional): Only used with Random effects. 
        Defaults to None.
    """
    
    # Unpack the labels
    label_y, label_x = labels
    
    # Create table, using the label for x to get a variable's coefficient,
    # standard error and t_value.
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i],
            results.get('t_values')[i],
            results.get('se_robust')[i],
            results.get('t_values_robust')[i]
        ]
        table.append(row)
    
    # Print the table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print extra statistics of the model.
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma2').item():.3f}")
    if _lambda: 
        print(f'\u03bb = {_lambda.item():.3f}')


def perm( Q_T: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.ndarray): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.ndarray): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    M,T = Q_T.shape 
    N = int(A.shape[0]/T)
    K = A.shape[1]

    # initialize output 
    Z = np.empty((M*N, K))
    
    for i in range(N): 
        ii_A = slice(i*T, (i+1)*T)
        ii_Z = slice(i*M, (i+1)*M)
        Z[ii_Z, :] = Q_T @ A[ii_A, :]

    return Z

def lag(T):
    I = np.identity(T) # Identity matrix
    low_triangular = np.tril(np.ones(T)) # Tx1 vector
    L_T = I - la.inv(low_triangular)
    return L_T[1:T]


def lead(T):
    I = np.identity(T) # Identity matrix
    upper_triangular = np.triu(np.ones(T)) # Tx1 vector
    L_T = I - la.inv(upper_triangular)
    return L_T[0:T-1,]

def demeaning_matrix(T):
    I = np.identity(T) # Identity matrix
    J = np.ones(T).reshape(-1,1) # Tx1 vector
    Q_T = I - J@la.inv(J.T@J)@J.T # p. 303
    return Q_T

def load_example_data():
    # First, import the data into numpy.
    data = np.loadtxt('wagepan.txt', delimiter=",")
    id_array = np.array(data[:, 0])

    # Count how many persons we have. This returns a tuple with the 
    # unique IDs, and the number of times each person is observed.
    unique_id = np.unique(id_array, return_counts=True)
    T = int(unique_id[1].mean())
    year = np.array(data[:, 1], dtype=int)

    # Load the rest of the data into arrays.
    y = np.array(data[:, 8]).reshape(-1, 1)
    x = np.array(
        [np.ones((y.shape[0])),
            data[:, 3],
            data[:, 9],
            data[:, 7],
            data[:, 5],
            data[:, 6],
            data[:, 4],
            data[:, 2]]
    ).T

    # Lets also make some variable names
    label_y = 'Log wage'
    label_x = [
        'Constant', 
        'Experience', 
        'Experience sqr', 
        'Union',
        'Married', 
        'Education', 
        'Hispanic', 
        'Black', 
    ]
    return y, x, T, year, label_y, label_x
