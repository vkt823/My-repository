import numpy as np
import ipywidgets as widgets # Interactive plots
import matplotlib.pyplot as plt
from types import SimpleNamespace


#Analytical solution, i.e. section 2
def consumption(alpha,beta,A, L=24): 
    """
        
    Args:

        alpha: relative preferences for consumption relative to leisure
        beta: output elasticity
        A: TFP
        L: Total time endowment. The default value is set to 24 to represent hours in the day.
        
    Returns:
    
        p = relative price of consumption good (Wage is normalized to 1)
        h = working hours (labor demand)
        y = production
        pi = profit
        Inc = income for consumer
        c = consumption
        l = leisure 
        
    """
    # a. Calculating the relative price that clear the goods and the labor market
    numerator=(L*alpha)**(1-beta)
    denominator = A*(beta**(beta/(1-beta))*(1-alpha)+alpha*beta**(1/(1-beta)))**(1-beta)
    p = numerator/denominator

    # b. Calculate labor demand, production and profit for the firm given the price in equilibrium
    h = (beta*p*A)**(1/(1-beta))
    y = A*(h)**beta
    pi = p*y - h

    # c. Define income, leisure and consumption
    Inc = pi+L
    c = alpha*Inc/p
    l = (1-alpha)*Inc

    # d. Check that good and labor market clear
    assert np.isclose(c, y,1e-8), 'Good market does not clear'
    assert np.isclose(L-h, l,0.0), 'Labor market does not clear'

    return p,h,y,pi,Inc,c,l

#Making an interactive plot of the solution
def interactive_figure(beta,A):
    """
        
    Args:

        beta: output elasticity
        A: TFP
        
    Returns:
    
        p_vec (in form of a figure)= vector of relative price of consumption good (Wage is normalized to 1) w.r.t. alpha
        c_vec (in form of a figure)= vector of consumption w.r.t. alpha
        l_vec (in form of a figure)= vector of leisure w.r.t. alpha
        
    """
    # a. Create vectors for x-axis and y-axis
    alpha_vec = np.linspace(1e-8,1-1e-8,10) #Cannot be 0 or 1 
    p_vec = np.empty(len(alpha_vec))

    # b. Solving the model for the given value of alpha
    for i, alpha in enumerate(alpha_vec):
        p_vec[i] = consumption(alpha,beta,A)[0]
    
    # c. Plotting the price over alpha
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_vec, p_vec, label='Price of consumption relative to leisure')
    ax.set_xlim([0.05,0.95]) # 
    ax.set_ylim([0,10]) #
    ax.set_title("Price")
    ax.set_xlabel("Alpha") 
    ax.legend(loc= 'upper right')

    plt.tight_layout()