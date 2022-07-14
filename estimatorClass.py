import numpy as np

class SF:
    def __init__(self, theta, alpha, variance):
        self.theta = theta
        self.alpha = alpha
        self.variance = variance
    
    def exp(self, theta, alpha):
        X = np.random.exponential(theta) #np.random.exponential is exp(1/theta)*1/theta
        S = (1/theta)*((X/theta)-1)
        return S * (X-alpha)**2
    
    def norm(self, theta, alpha, variance):
        X = np.random.normal(theta, variance)
        return (1/variance) * (X - theta) * (X - alpha)**2
    
    def mining(self, theta):
        X = np.random.exponential(1/theta)
        S = (1/theta)*((X/theta)-1)
        h = X/np.sqrt(1+X)
        return 1 - h * S

class IPA:
    def __init__(self, theta, alpha, variance):
        self.theta = theta
        self.alpha = alpha
        self.variance = variance
    
    def norm(self, theta, alpha, variance):
        return (np.random.normal(theta, variance) - alpha) *  2 
    
    def exp(self, theta, alpha):
        X = np.random.exponential(theta) #np.random.exponential is exp(1/theta)*1/theta
        return (2/theta) * X * (X - alpha)
    
    def mining(self, theta):
        X = np.random.exponential(1/theta)
        return 1 - (1/theta)*(X/np.sqrt(X+1)) - (1/(2*theta))*((X**2))/(((X+1)*np.sqrt(X+1)))
        

def boundary(x, delta):
    """ We require delta < 1. """
    if x <= delta:
        return delta
    elif x >= 1 / delta:
        return 1 / delta
    else: 
        return x      