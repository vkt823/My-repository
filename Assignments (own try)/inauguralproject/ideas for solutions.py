#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Question 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!IDEA 1 WITH CLASSES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class Human():
    
    def __init__(self,name,height,weight): # called when created (initialize). All methods start with the argument self.
        #CHR: the idea is to create variables alpha and sigma in the init
        
        # save the inputs as attributes
        self.name = name # an attribute
        self.height = height # an attribute
        self.weight = weight # an attribute
    
    def bmi(self): # defines a method
        
        bmi = self.weight/(self.height/100)**2 # calculate bmi
        return bmi # output bmi
    
    def print_bmi(self): #defines another method
        print(self.bmi()) 


# a. create an instance of the human object called "jeppe"        
jeppe = Human('jeppe',182,80) # height=182, weight=80
print(type(jeppe))

# b. print an attribute
print(jeppe.height)

# c. print the result of calling a method
print(jeppe.bmi()) #when you call a method then you need the brackets



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Idea 2 with classes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Attributes can be changed and extracted with.-notation

jeppe.height = 160
print(jeppe.height)
print(jeppe.bmi())

#Or with setattr- and getaattr-notation (set attribute/get attribute)

setattr(jeppe,'height',182) # jeppe.height = 182. Changes the attribute.
height = getattr(jeppe,'height') # height = jeppe.height. Gets the attribute out of the object.
print(height)
print(jeppe.bmi())


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Idea 3 with classes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class Agent:
    def __init__(self,**kwargs): 
        #CHR: **kwargs instead of specifying alpha and sigma in init definition
        
        self.name = 'Asker'

        self.alpha = 0.5
        
        self.p1 = 1
        self.p2 = 2
        self.I = 10
        
        self.x1 = np.nan # not-a-number - we dont have a solution yet
        self.x2 = np.nan

        self.solved = False

        for key, value in kwargs.items():
            setattr(self,key,value) # like self.key = value

        #CHR: need this last for loop to make kwargs work


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!print of class!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#CHR: Can use these methods for printing

    def __str__(self):
        '''
        Called when print() is called 
        Ignore this for now, it simply prints a lot of information about the class
        '''

        # f-strings (f'') allows for including the string version of variables inside a string if they are inside {}-brackes
        # They also allow for choosing how many digits are printed, we'll learn more about them later
        text = f'The Agent is called {self.name} and faces this problem: \n' 
        text += f'Income = {self.I} \n' # \n is line break
        text += f'\u03B1 = {self.alpha} \n' # \u03B1 is alpha in unicode for printing
        text += f'Prices are (p1,p2) = ({self.p1},{self.p2})\n'
        if self.solved: # Checks if agent problem has been solved
            text += f'Optimal consumption of (x1,x2) = ({self.x1},{self.x2})'
        else:
            text += 'Optimal consumption of (x1,x2) has not been found'
        return text

    def print_solution(self):
        # Prints the solution and the utility and expenditure
        text = f'Optimal consumption of (x1,x2) = ({self.x1},{self.x2}) \n'
        text += f'Utility is {self.u_func(self.x1,self.x2):.2f}\n'
        text += f'Expenditure is {self.expenditure(self.x1,self.x2)}'
        print(text)



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!output!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#SHouldn't we output with the sol vectors??