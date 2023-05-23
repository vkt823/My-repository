import platformdirs
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

class numerical_solution():
    
    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. parameters
        par.alpha = 0.5
        par.beta = 0.5
        par.A = 20
        par.L = 75

        # b. solution
        sol = self.sol = SimpleNamespace()

    def production_function(self,h):
        "production function of the firm"

        #a. unpack
        par = self.par

        #b. production function
        y = par.A*h**par.beta
        
        #c. output
        return y

    def firm_profit(self,h,p):
        "profit function of the firm"

        #a. profit
        pi = p*self.production_function(h)-h

        #b. output
        return -pi
    
    def firm_profit_maximization(self,p):

        #a. unpack
        par = self.par
        sol = self.sol

        #b. call optimizer
        bound = ((0,par.L),)
        x0=[0.0]
        sol_h = optimize.minimize(self.firm_profit,x0,args = (p,),bounds=bound,method='L-BFGS-B')

        #c. unpack solution
        sol.h_star = sol_h.x[0]
        sol.y_star = self.production_function(sol.h_star)
        sol.pi_star = p*sol.y_star-sol.h_star

        return sol.h_star, sol.y_star, sol.pi_star

    def utility(self,c,h):
        "utility of the consumer"

        #a. unpack
        par = self.par

        #b. utility
        u = c**par.alpha*(par.L-h)**(1-par.alpha)

        #c. output
        return u

    def income(self,p):
        "consumer's income/budget constraint"

        #a. unpack
        par = self.par
        sol = self.sol

        #b. budget constraint. Minus because the income is defined negatively in order to optimize. 


        h_inc,y_inc,pi_inc=self.firm_profit_maximization(p)

        sol.Inc = pi_inc+par.L

        #c. output
        return sol.Inc


    def maximize_utility(self,p):
        
        # a.unpack
        par = self.par
        sol = self.sol

        # a. solve using standard solutions
        utility_inc=self.income(p)

        sol.c_star = par.alpha*utility_inc/p
        sol.l_star = (1-par.alpha)*utility_inc

        return sol.c_star, sol.l_star


    def utility_maximization(self,p): 

        #a. unpack
        par = self.par
        sol = self.sol

        #b. call optimizer
        #Bounds
        bounds = ((0,np.inf),(0,par.L))
        #Initial guess
        x0=[25,8]

        #Constraints. The income must be equal to or greater than the income. We first define l 
        constraint = sol.Inc-p*self.utility[0]-par.L-sol.h_star
        ineq_con = {'type': 'ineq', 'fun': constraint} 


        # b. call optimizer
        sol_con = optimize.minimize(self.utility,x0,
                             method='SLSQP',
                             bounds=bounds,
                             constraints=[ineq_con],
                             options={'disp':True})
        c_star = sol_con.x[0]
        l_star = sol_con.x[1]

        return c_star, l_star
    
    def market_clearing(self,p):
        "calculating the excess demand of the good and working hours"
        #a. unpack
        par = self.par
        sol = self.sol

        #b. optimal behavior of firm
        h,y,pi=self.firm_profit_maximization(p)

        #c. optimal behavior of consumer
        c,l=self.maximize_utility(p)

        #b. market clearing
        goods_market_clearing = y - c
        labor_market_clearing = h - par.L + l

        return goods_market_clearing, labor_market_clearing
    
    def find_relative_price(self,tol=1e-4,iterations=500, p_lower=0.25, p_upper=0.75):
        "find price that causes markets to clear"

        #Initial values.                                                                                                       
        i=0

        while i<iterations: 
            
            p=(p_lower+p_upper)/2
            f = self.market_clearing(p)[0]

            if np.abs(f)<tol: 
                good_clearing=self.market_clearing(p)[0]
                labor_clearing=self.market_clearing(p)[1]
                print(f' Step {i:.2f}: p = {p:.2f} -> {f:12.8f}. Good clearing = {good_clearing:.2f}. Labor clearing = {labor_clearing:.2f}. ')
                break
            elif self.market_clearing(p_lower)[0]*f<0:
                p_upper=p
                #print(f' Step {i:.2f}: p = {p:.2f} -> {f:12.8f}')
            elif self.market_clearing(p_upper)[0]*f<0:
                p_lower=p
                #print(f' Step {i:.2f}: p = {p:.2f} -> {f:12.8f}')
            else: 
                print("Fail")
                return None
            
            i+=1
        return p, good_clearing, labor_clearing