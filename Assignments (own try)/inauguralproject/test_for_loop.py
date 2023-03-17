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

        par.alpha_vec = [0.25, 0.50, 0.75]
        par.sigma_vec = [0,0.5,1]

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size) #Hours worked by man
        sol.HM_vec = np.zeros(par.wF_vec.size) #wage by man
        sol.LF_vec = np.zeros(par.wF_vec.size) #hours by female
        sol.HF_vec = np.zeros(par.wF_vec.size) #wage female

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        H = np.zeros(len(par.alpha_vec)*len(par.sigma_vec))
        for i,a in enumerate(par.alpha_vec):
            for j,s in enumerate(par.sigma_vec):
                if s == 0:
                    H[i*len(par.sigma_vec)+j] = np.fmin(HM[i*len(par.sigma_vec)+j],HF[i*len(par.sigma_vec)+j])
                elif s == 1:
                    H[i*len(par.sigma_vec)+j] = HM[i*len(par.sigma_vec)+j]**(1-a)*HF[i*len(par.sigma_vec)+j]**a
                else:
                    H[i*len(par.sigma_vec)+j] = ((1-a)*HM[i*len(par.sigma_vec)+j]**((s-1)/s)+a*HF[i*len(par.sigma_vec)+j]**((s-1)/s))**(s/(s-1))
        
        H = H.reshape(len(par.alpha_vec), len(par.sigma_vec), HM.shape[0])


        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)
        #CHR: this is the first part of the utility function


        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        #CHR: this is the second part of the utility function
        
        return utility - disutility
        #Chr: returns the utility function

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49) #X takes a value for every half an hour of the day
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
        LM = LM.ravel() 
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = np.zeros((len(par.alpha_vec), len(par.sigma_vec), LM.shape[0]))
        for i,a in enumerate(par.alpha_vec):
            for j,s in enumerate(par.sigma_vec):
                H = self.calc_home_production(HM[i*len(par.sigma_vec)+j], HF[i*len(par.sigma_vec)+j], a, s)
                C = par.wM*LM + par.wF*LF
                Q = C**par.omega*H**(1-par.omega)
                utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)
                # d. disutility of work
                epsilon_ = 1+1/par.epsilon
                TM = LM+HM[i*len(par.sigma_vec)+j]
                TF = LF+HF[i*len(par.sigma_vec)+j]
                disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
                u[i,j,:] = utility - disutility
        u = u.reshape(len(par.alpha_vec)*len(par.sigma_vec), LM.shape[0])

        # c. set to minus infinity if constraint is broken
        I = (LM+HM.repeat(len(par.alpha_vec)*len(par.sigma_vec)) > 24) | (LF+HF.repeat(len(par.alpha_vec)*len(par.sigma_vec)) > 24)
        u[:,I] = -np.inf

        # d. find maximizing argument
        j = np.argmax(u)
        #CHR: gemmer index for bedste utility
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]
        #indsætter index for bedste utility

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt