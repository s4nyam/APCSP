
# Imports
from Board import Board
from Kernel import Kernel
from Growth_fn import Growth_fn
from Automaton import Automaton

import numpy as np
from matplotlib import pyplot as plt
import time
import warnings
warnings.simplefilter("ignore", UserWarning)

board_initialisation = 'random'
# zeros ones random sparse gaussian ring

kernel_type = 'circular_kernel'
# square_kernel circular_kernel ring_kernel smooth_ring_kernel kernel_shell

sigma = 0.5
mu = 0.25
dt = 0.1

kernel = None
board = None
board_size = 64
kernel_size = 16
frames = 100
seed = None
kernel_peaks = None


class Lenia:
    def __init__(self):
        self.board = board
        self.board_size = board_size
        self.seed = seed
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.frames = frames
        self.kernel_size = kernel_size
        self.kernel_peaks = kernel_peaks
        self.kernel = kernel
        self.board_initialisation_type = board_initialisation
        self.kernel_type = kernel_type

    def run_simulation(self) -> None:
 

        # Kernel
        kernel_size = 16
        try:
            if self.kernel_size != None: kernel_size = int(self.kernel_size)
        except: pass
        # kernel peak refers to the maximum value of a convolutional kernel
        kernel_peaks = np.array([1])
        try:
            if self.kernel_size != None: kernel_peaks = np.array([float(i) for i in self.kernel_size])
        except: pass
            
        # if(self.kernel == None): 
        # else: kernel = self.kernel
        if(self.kernel == None):
            if(self.kernel_type == 'square_kernel'): kernel = Kernel().square_kernel(3,1)
            elif(self.kernel_type == 'circular_kernel'): kernel = Kernel().circular_kernel(kernel_size) # Create kernel
            elif(self.kernel_type == 'ring_kernel'): kernel = Kernel().ring_kernel(kernel_size, kernel_size // 2) # Create kernel
            elif(self.kernel_type == 'smooth_ring_kernel'): kernel = Kernel().smooth_ring_kernel(diameter=kernel_size) # Create kernel
            else: kernel = Kernel().kernel_shell(kernel_size, peaks=kernel_peaks) # Create kernel
        else:
            kernel = self.kernel
   
        
        
        # Growth fn
        growth_fn = Growth_fn()
        
        if self.mu != None: growth_fn.mu = self.mu
        if self.sigma != None: growth_fn.sigma = self.sigma
        
        # Board  
        board_size = 64
        if self.board_size != None: board_size = int(self.board_size)
        
        seed = None
        try: seed = int(self.seed)
        except: pass
        
        board = Board(board_size, seed=seed) # Create board
        board.initialisation_type = self.board_initialisation_type
        try: 
            if self.board != None: 
                board.board = self.board # if provided
        except: pass

        # General simulation params
        frames = 100
        try: frames = int(self.frames)
        except: pass
        
        dt = 0.1 # timestep
        if self.dt != None: dt = float(self.dt)
        
        # Run the simulation and animate
        # print('Running simulation... ')
        handlerCA = Automaton(board, kernel, growth_fn, dT=dt,kernel_type=self.kernel_type)

        # it calls update_convolution which call growth function 
        handlerCA.animate(frames)
        # print('Simulation complete!')
        # timestr = time.strftime("%Y%m%d%H%M%S")
        outfile = 'output_'+str(board_initialisation)+"_"+str(kernel_type)+"_mu"+str(mu)+"_sigma"+str(sigma)+'.gif'   
        print('./outputs/{}...)'.format(outfile))
        handlerCA.save_animation(outfile)
        handlerCA.plot_kernel_info(save=True)
        # lenia_board_state = handlerCA.save_recorded_board_state(timestr+".txt")
        # return self.process_board_state(lenia_board_state)




if __name__ == "__main__":
    
    lenia = Lenia()
    lenia.run_simulation()