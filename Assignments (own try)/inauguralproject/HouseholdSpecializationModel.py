from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:
    
    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)
        par.log_wF_vec = np.log(np.linspace(0.8,1.2,5))

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production 
        #Based on the value of sigma we use the specified function for H
        if par.sigma == 0:
            H=np.fmin(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H= ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+
                par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]
        opt.HF_HM = HF[j]/HM[j] #Calculate ratio

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_wF_vec(self, discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        # We create a vector to store the log ratio of HF/HM for the different log ratios of wF/wM
        log_HF_HM = np.zeros(par.wF_vec.size)

        # We loop over each value of wF in wF_vec
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF # Set the new value of wF
            
            # Solve the model
            if discrete==True:
                opt = self.solve_discrete()
                log_HF_HM[i] = np.log(opt.HF_HM)
            else:
                opt = self.solve_continously()
                log_HF_HM[i] = np.log(opt.HF_HM)

        par.wF = 1.0
        #We return wF to original value

        return log_HF_HM #Return the vector of log ratio of HF/HM
    
    #We need negative value of utility function. 
    def utility_function(self, L): 
        return -self.calc_utility(L[0],L[1],L[2],L[3])
    
    def solve_continously(self):
        #Calling the values from previous
        par = self.par
        opt = SimpleNamespace()
        

        #Define the bounds and constraints. 
        constraint_men = ({'type': 'ineq', 'fun': lambda L:  24-L[0]-L[1]})
        constraint_women = ({'type': 'ineq', 'fun': lambda L:  24-L[2]-L[3]})
        bounds=((0,24),(0,24), (0,24), (0,24))
        
        # Initial guess. Not important
        initial_guess = [12,12,12,12]

        # Call optimizer
        solution_cont = optimize.minimize(
        self.utility_function, initial_guess,
        method='SLSQP', bounds=bounds, constraints=(constraint_men, constraint_women))
        
        # Save results
        opt.LM = solution_cont.x[0]
        opt.HM = solution_cont.x[1]
        opt.LF = solution_cont.x[2]
        opt.HF = solution_cont.x[3]
        opt.HF_HM = solution_cont.x[3]/solution_cont.x[1] #calculate ratio
        
        return opt

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass