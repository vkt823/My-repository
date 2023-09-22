import numpy as np
from numpy import linalg as la
from tabulate import tabulate



def estimate(y: np.ndarray, x: np.ndarray, transform='', N=None, T=None) -> list:
    """Takes some np.arrays and estimates regular OLS, FE or FD.
    

    Args:
        y (np.ndarray): The dependent variable, needs to have the shape (n*t, 1)
        x (np.ndarray): The independent variable(s). If only one independent 
        variable, then it needs to have the shape (n*t, 1).
        transform (str, optional): Specify if estimating fe or fd, in order 
        to get correct variance estimation. Defaults to ''.
        N (int, optional): Number of observations. If panel, then the 
        number of individuals. Defaults to None.
        T (int, optional): If panel, then the number of periods an 
        individual is observerd. Defaults to None.

    Returns:
        dict: A dictionary with the results from the ols-estimation.
    """
    
    b_hat = est_ols(y, x)
    resid = y - x@b_hat
    SSR = resid.T@resid
    SST = (y - np.mean(y)).T@(y - np.mean(y))
    R2 = 1 - SSR/SST

    sigma, cov, se = variance(transform, SSR, x, N, T)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma, t_values, R2, cov]
    return dict(zip(names, results))

    
def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates OLS using input arguments.

    Args:
        y (np.ndarray): Check estimate()
        x (np.ndarray): Check estimate()

    Returns:
        np.array: Estimated beta hats.
    """
    return la.inv(x.T@x)@(x.T@y)
    # alternatively and more efficiently: 
    # return la.solve(x.T@x, x.T@y)

def variance( 
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        N: int,
        T: int
    ) -> tuple :
    """Use SSR and x array to calculate different variation of the variance.

    Args:
        transform (str): Specifiec if the data is transformed in any way.
        SSR (float): SSR
        x (np.ndarray): Array of independent variables.
        N (int, optional): Number of observations. If panel, then the 
        number of individuals. Defaults to None.
        T (int, optional): If panel, then the number of periods an 
        individual is observerd. Defaults to None.

    Raises:
        Exception: [description]

    Returns:
        tuple: [description]
    """

    # Number of coefficients
    K=x.shape[1]

    if transform in ('', 're', 'fd'):
          sigma = SSR/(N*T-K)
    elif transform.lower() == 'fe':
          sigma = SSR/(N*T-N-K) 
    elif transform.lower() in ('be'): 
          sigma = SSR/(N-K) 
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma, cov, se


def print_table(labels: tuple, results: dict, headers=None, title="Results", **kwargs) -> None:
    """
    The `print_table` function takes in labels, results, headers, and title as arguments, and prints a
    table with the results and data for model specification.
    
    Args:
      labels (tuple): The `labels` parameter is a tuple that contains the labels for the dependent
    variable and the independent variables. The first element of the tuple is the label for the
    dependent variable, and the remaining elements are the labels for the independent variables.
      results (dict): The `results` parameter is a dictionary that comes from the `estimate` function.
      headers: The `headers` parameter is a list that specifies the column headers for the table. By
    default, it is set to `["", "Beta", "Se", "t-values"]`.
      title: The title of the table, which is "Results" by default. Defaults to Results
    """

    # Unpack labels
    label_y, label_x = labels
    
    # Default headers if not provided
    if headers is None:
        headers = ["", "Beta", "Se", "t-values"]

    # Create table for data on coefficients
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print data for model specification
    print(f"R² = {results.get('R2').item():.3f}")
    print(f"σ² = {results.get('sigma').item():.3f}")
    
    
def perm( Q_T: np.ndarray, A: np.ndarray, t=0) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.array): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.array): The vector or matrix that is to be transformed. Has
        to be a 2d array.
        
        t (int, optional): The number of years an individual is in the sample.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer T from the shape of the transformation matrix.
    if t==0:
        t = Q_T.shape[1]

    # Initialize the numpy array
    Z = np.array([[]])
    Z = Z.reshape(0, A.shape[1])

    # Loop over the individuals, and permutate their values.
    for i in range(int(A.shape[0]/t)):
        Z = np.vstack((Z, Q_T@A[i*t: (i + 1)*t]))
    return Z
