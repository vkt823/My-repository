
def all(model):

    model.y = 2*model.x
    #CHR: Inside the function, a new variable "y" is 
    # defined in the "model" object as twice the value 
    # of "x" attribute of "model". Then, a string is 
    # printed using an f-string that includes the value 
    # of "model.x".
    print(f'solving - with {model.x = :}')