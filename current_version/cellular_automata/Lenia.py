
# Imports
from cellular_automata.Board import Board
from cellular_automata.Kernel import Kernel
from cellular_automata.Growth_fn import Growth_fn
from cellular_automata.Automaton import Automaton

import argparse
import numpy as np
from matplotlib import pyplot as plt
import json
import time
import warnings
warnings.simplefilter("ignore", UserWarning)

NUMERIC_ARGS = ['board_size', 'kernel_size', 'dt', 'frames', 'sigma', 'mu', 'seed']
OPTIONAL_ARGS = ['board_size', 'kernel_size', 'dt', 'frames', 'sigma', 'mu', 'seed', 'b1', 'b2', 's1', 's2']


class Lenia:
    def __init__(self, args):
        self.board = args["board"]
        self.board_size = args["board_size"]
        self.seed = args["seed"]
        self.mu = args["mu"]
        self.sigma = args["sigma"]
        self.dt = args["dt"]
        self.frames = args["frames"]
        self.kernel_size = args["kernel_size"]
        self.kernel_peaks = args["kernel_peaks"]


    def run_simulation(self) -> None:
        """Run and save a simulation using the generalised Lenia framework for a given set of parameters
        Parameters may include:
        - The initial values of the board
            - or board size / seed to create a board
        - The kernel 
            or kernel peaks / kernel size to create a kernel
        - The growth function type (Gaussian/Bosco) and corresponding parameters
        - The value of dT
        - The number of frames to simulate and output format

        Args:
            d_data (dict): The parameters 
        """
        

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
            
        kernel = Kernel().kernel_shell(kernel_size, peaks=kernel_peaks) # Create kernel
        try: kernel = self.kernel # use kernel provided (if exists)
        except: pass  
        
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
        print('Running simulation... ')
        game_of_life = Automaton(board, kernel, growth_fn, dT=dt)

        # it calls update_convolution which call growth function 
        game_of_life.animate(frames)
        print('Simulation complete!')

        outfile = 'output.gif'
       
        print('Saving simulation as ./outputs/{}... (may take several minutes)'.format(outfile))
        game_of_life.save_animation(outfile)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        lenia_board_state = game_of_life.save_recorded_board_state(timestr+".txt")
        return self.process_board_state(lenia_board_state)
        
    def validate_args(args:dict) -> dict:
        """Check the arguments provided by the user are valid. Return arguments as correct type.

        Args:
            args (dict): CLI args from user

        Returns:
            dict: Cleaned and checked args
        """
        
        # Convert numeric args to floats
        for arg in NUMERIC_ARGS:
            
            if args[arg] != None:
                try:
                    args[arg] = float(args[arg])
                except ValueError:
                    print('ERROR: --{} must be a numeric value'.format(arg.replace('_','-')))
                    return(-1)
                if args[arg] < 0:
                    print('ERROR: --{} must be greater then zero'.format(arg.replace('_','-')))
                    return(-1)
        
        # Check the kernel peaks are all numeric
        if args['kernel_peaks'] != None:
            try:
                args['kernel_peaks'] = [float(i) for i in args['kernel_peaks']]
            except ValueError:
                print('ERROR: --kernel-peaks must be a numeric')
                return(-1)
            for i in args['kernel_peaks']:
                if i < 0 or i > 1:
                    print('ERROR: --kernel-peaks must be between 0 and 1')
                    return(-1)
    
        return args
            


    def process_board_state(self, lenia_board_state):
        live_cell_count_in_each_frame = {}
        i = 0
        sum = 0
        for board_state in lenia_board_state:
            i +=1
            board_state_dict = json.loads(board_state)
            board_arr = np.array(board_state_dict['board'])
            board_arr = board_arr.flatten()
            board_arr_greater_than_zero = list(board_arr[board_arr > 0.5])
            live_cell_count_in_each_frame["frame"+str(i)] = len(board_arr_greater_than_zero)
            sum += len(board_arr_greater_than_zero)
        live_cell_count_in_each_frame["live_cells"] = sum
        return live_cell_count_in_each_frame


    