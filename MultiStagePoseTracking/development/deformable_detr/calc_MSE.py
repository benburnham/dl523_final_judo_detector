import numpy as np

def calc_MSE(x1,y1,x2,y2):
    return np.sqrt(((x1+x2)/2-640)**2 + ((y1+y2)/2-360)**2)

