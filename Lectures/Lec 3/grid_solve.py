# All modules used within a module must be imported locally
import numpy as np

# You need to respecify the u_func, because the module does not share scope with the notebook. 
# That is, the module functions cannot see that u_func was defined in the notebook when find_best_choice is called
def u_func(x1,x2,alpha=0.50):
    return x1**alpha * x2**(1-alpha)

def find_best_choice(alpha,I,p1,p2,N1,N2,do_print=True):
    
    # a. allocate numpy arrays
    shape_tuple = (N1,N2) #CHR: we create a tuple, like a grid
    x1_values = np.empty(shape_tuple) #CHR: we create empty vector x1
    x2_values = np.empty(shape_tuple) #CHR: we create empty vector x2
    u_values = np.empty(shape_tuple) #CHR: we create empty vector of utility
    
    # b. start from guess of x1=x2=0
    #CHR: we start by telling python "our best guess" - we just need any initial guess
    x1_best = 0
    x2_best = 0
    u_best = u_func(0,0,alpha=alpha)
    
    # c. loop through all possibilities
    #we loop throgh all the elements in the grid -> both values for x1 and x2 -> double for loop
    for i in range(N1): #loops through rows
        for j in range(N2): #loops through columns
            
            # i. x1 and x2 (chained assignment)
            x1_values[i,j] = x1 = (i/(N1-1))*I/p1 #we're just counting upwards, can be interpreted as a percentage of the maxpoint, which here is I
            x2_values[i,j] = x2 = (j/(N2-1))*I/p2
            #CHR: when we write x1_values[i,j] we're assigning value for x1 for the space i,j in the grid
            #CHR: Chained assignment - we're both filling out the numpy array x1_values,
            # and then we're locally inside the loop creating the variable x1

            # ii. utility
            if p1*x1 + p2*x2 <= I: # u(x1,x2) if expenditures <= income 
                u_values[i,j] = u_func(x1,x2,alpha=alpha) #CHR: then we can just put in the utility as the utility
            else: # u(0,0) if expenditures > income, not allowed
                u_values[i,j] = u_func(0,0,alpha=alpha) #CHR: then we're spending 0, and then we know it will not be the best guess 
            
            # iii. check if best sofar
            if u_values[i,j] > u_best:
                x1_best = x1_values[i,j]
                x2_best = x2_values[i,j] 
                u_best = u_values[i,j]
                #CHR: if the given u_value is better than the one so far, then we store them as our new values
    
    # d. print
    if do_print:
        print_solution(x1_best,x2_best,u_best,I,p1,p2)

    return x1_best,x2_best,u_best,x1_values,x2_values,u_values
    #CHR: we return the solution and the grid of values of all the possible solution
    #CHR: when we return it this way we implicitly define a tuple
# function for printing the solution
def print_solution(x1,x2,u,I,p1,p2):
    print(f'x1 = {x1:.4f}')
    print(f'x2 = {x2:.4f}')
    print(f'u  = {u:.4f}')
    print(f'I-p1*x1-p2*x2 = {I-p1*x1-p2*x2:.8f}')
    print(f'x1*p1/I = {x1*p1/I:.4f}')


def find_best_choice_monotone(alpha,I,p1,p2,N,do_print=True):
    
    # a. allocate numpy arrays
    shape_tuple = (N)
    x1_values = np.empty(shape_tuple)
    x2_values = np.empty(shape_tuple)
    u_values = np.empty(shape_tuple)
    
    # b. start from guess of x1=x2=0
    x1_best = 0
    x2_best = 0
    u_best = u_func(0,0,alpha)
    
    # c. loop through all possibilities
    for i in range(N):
        #CHR: Note we only need one loop now because we're just looking for possible solutions of x1
        #and then we use monotonicity to say that the individual spends the rest on x2
        
        # i. x1
        x1_values[i] = x1 = i/(N-1)*I/p1
        
        # ii. implied x2 by budget constraint
        x2_values[i] = x2 = (I-p1*x1)/p2
        #CHR: here is the monotonicity assumption
        #CHR: Note we no longer need to write our budget constraint, because it is implicite in x2 that we spend exactly our income
            
        # iii. utility    
        u_values[i] = u_func(x1,x2,alpha)
        
        if u_values[i] >= u_best:    
            x1_best = x1_values[i]
            x2_best = x2_values[i] 
            u_best = u_values[i]
            
    # d. print
    if do_print:
        print_solution(x1_best,x2_best,u_best,I,p1,p2)   

    return x1_best,x2_best,u_best,x1_values,x2_values,u_values