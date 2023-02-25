
import numpy as np

class Growth_fn(object):
    """Class for the growth function which is used to update the board based on the neighbourhood sum.
    This replaces the traditional conditional update used in Conway's game of life and can be generalised to any
    continous function. 
    
    f(x,y,t+1) = g(k*f(x,y,t))
    
    where g is the growth function
    k is the update kernel 
    f(x,y,t) is the board state at time t
    N.b. The operator * is the convolution operator 
    
    It consists of growth and shrink parts, which act on the neighbourhood sum to update the board at each timestep.
    """
    def __init__(self):
        
        
        # Values for Gaussian update rule
        self.mu = 0.135
        self.sigma = 0.015
        self.growth_fn = self.growth_gaussian


 
    def growth_gaussian(self, U:np.array) -> np.array:
        """Use a smooth Gaussian growth function to update the board, based on the neighbourhood sum.
        This is the function used by Lenia to achive smooth, fluid-like patterns.

        Args:
            U (np.array): The neighbourhood sum 

        Returns:
            np.array: The updated board at time t = t+1
        """
        gaussian = lambda x, m, s: np.exp(-( (x-m)**2 / (2*s**2) ))
        return gaussian(U, self.mu, self.sigma)*2-1