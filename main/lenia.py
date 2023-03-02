
# Imports
# https://chakazul.github.io/Lenia/JavaScript/Lenia.html
import numpy as np
from matplotlib import pyplot as plt
import time
import warnings
warnings.simplefilter("ignore", UserWarning)
#Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.signal
import os.path
import os
from datetime import datetime
from json import JSONEncoder
import json
from pathlib import Path
                

OUTPUT_PATH = './outputs'
MAX_FRAMES = 3000

mu = 0.31
sigma = 0.057
dt = 0.1

kernel_size = 16
board_size = 64

frames = 120
seed = None
kernel_peaks = np.array([1])
kernel_diameter = 16
frame_intervals = float(50)
def rps_glider1(m, n):
        pop = np.zeros([m, n], dtype = int)
        pop[2, 2:7] = 1
        pop[3, 2] = 1
        pop[3, 3:6] = 2
        pop[3, 6] = 1
        pop[4, 2] = 1
        pop[4, 3:6] = 2
        pop[4, 6] = 1
        pop[5, 2] = 1
        pop[5, 3:6] = 2
        pop[6, 2:5] = 1
        return pop


# INTERESTING KERNEL
def spider_web_kernel( 
                        diameter:int, 
                    peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                    kernel_mu:float=0.5, 
                    kernel_sigma:float=0.15, 
                    a:float=4.0):
    m=100
    n=100
    # create a grid with zeros
    grid = np.zeros((n, m))
    
    # calculate the center of the grid
    center_x = n // 2
    center_y = m // 2
    
    # create a meshgrid
    x, y = np.meshgrid(np.arange(n), np.arange(m))
    
    # calculate the distance of each point from the center
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # calculate the smoothing factor
    smoothing_factor = 0.5
    
    # calculate the values for each point
    grid = np.sin(distance * smoothing_factor) * distance
    
    return grid


class Lenia:
    def __init__(self, kernel):
        self.sigma = sigma
        self.mu = mu
        self.dt = dt
        self.kernel_size = kernel_size
        self.kernel_diameter = kernel_diameter
        self.kernel_peaks = kernel_peaks
        self.kernel = kernel
        self.normalise_kernel()
        self.board_size = board_size
        self.frames = frames
        self.seed = seed
        
        self.frame_intervals = frame_intervals
        self.anim = None
        self.lenia_board_state = {}
        
        
        # For random initialisation
        self.board = np.random.rand(self.board_size, self.board_size)
        
        # For single pixel in center initialisation
        # self.board = np.zeros((self.board_size, self.board_size))
        
        # For probabilistic initialisation
        prob = 0.05
        # self.board = np.random.choice([0, 1], size=(self.board_size, self.board_size), p=[1-prob, prob])

        # self.board[self.board_size//2, self.board_size//2] = 1
        self.cmap = 'viridis'
        self.fig, self.img = self.show_board()
        

    # KERNELS AND ITS TWEAKS - KERNELS BEGIN HERE
    # KERNELS AND ITS TWEAKS - KERNELS BEGIN HERE
    # KERNELS AND ITS TWEAKS - KERNELS BEGIN HERE
    # KERNELS AND ITS TWEAKS - KERNELS BEGIN HERE



    


    # KERNELS AND ITS TWEAKS - KERNELS END HERE
    # KERNELS AND ITS TWEAKS - KERNELS END HERE
    # KERNELS AND ITS TWEAKS - KERNELS END HERE
    # KERNELS AND ITS TWEAKS - KERNELS END HERE





    # FLEXIBLITY TO CHANGE GROWTH FUNCTION
    def growth_function1(self, U:np.array):
        gaussian = lambda x, m, s: np.exp(-( (x-m)**2 / (2*s**2) ))
        return gaussian(U, self.mu, self.sigma)*2-1


    def show_board(self, 
                   display:bool=False,
                   ):
        dpi = 50 # Using a higher dpi will result in higher quality graphics but will significantly affect computation

        self.fig = plt.figure(figsize=(10*np.shape(self.board)[1]/dpi, 10*np.shape(self.board)[0]/dpi), dpi=dpi)

        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        self.img = ax.imshow(self.board, cmap=self.cmap, interpolation='none', aspect=1, vmin=0) #  vmax=vmax
        
        if display:
            plt.show()
        else: # Do not show intermediate figures when creating animations (very slow)
            plt.close()

        return self.fig, self.img
    
    
    def animate(self):
        self.anim =  matplotlib.animation.FuncAnimation(self.fig, self.animate_step, 
                                            frames=self.frames, interval=self.frame_intervals, save_count=MAX_FRAMES, blit=True)

    
    def animate_step(self, i:int) -> plt.imshow:
        neighbours = scipy.signal.convolve2d(self.board, self.kernel, mode='same', boundary='wrap')
        self.board = np.clip(self.board + self.dt * self.growth_function1(neighbours), 0, 1)
        if 45<=i<55:
            # print("i: ", i)
            self.record_board_state(i)
        self.img.set_array(self.board) # render the updated state 
        return self.img,
    
    
    def save_animation(self, dir, 
                       filename:str,
                       ):
        if not self.anim:
            raise Exception('ERROR: Run animation before attempting to save')
            return 
        output_path = OUTPUT_PATH+"/"+dir
        Path(output_path).mkdir(parents=True, exist_ok=True)
        fmt = os.path.splitext(filename)[1] # isolate the file extension
        
        if fmt == '.gif':
            f = os.path.join(output_path, filename) 
            writer = matplotlib.animation.PillowWriter(fps=30) 
            self.anim.save(f, writer=writer)
        else:
            raise Exception('ERROR: Unknown save format. Must be .gif or .mp4')


    
    def normalise_kernel(self) -> np.array:

        kernel_norm = self.kernel / (1*np.sum(self.kernel))
        self.norm_factor = 1/ (1*np.sum(self.kernel))
        self.kernel = kernel_norm 
        return kernel_norm
        
        
    def plot_kernel_info(self,
                         dir,
                         cmap:str='viridis', 
                         bar:bool=False,
                         save:str=None,
                         ) -> None:

        
        k_xsection = self.kernel[self.kernel.shape[0] // 2, :]
        k_sum = np.sum(self.kernel)
        
        fig, ax = plt.subplots(1, 3, figsize=(14,2), gridspec_kw={'width_ratios': [1,1,2]})
        
        # Show kernel as heatmap
        ax[0].imshow(self.kernel, cmap=cmap, vmin=0)
        ax[0].title.set_text('Kernel')
        
        # Show kernel cross-section
        ax[1].title.set_text('Kernel Cross-section')
        if bar==True:
            ax[1].bar(range(0,len(k_xsection)), k_xsection, width=1)
        else:
            ax[1].plot(k_xsection)
        
        # Growth function
        ax[2].title.set_text('Growth Function')
        x = np.linspace(0, k_sum, 1000)
        ax[2].plot(x, self.growth_function1(x))
        
        if save:
            output_path = OUTPUT_PATH+"/"+dir
            Path(output_path).mkdir(parents=True, exist_ok=True)
            print('Saving kernel and growth function info to', os.path.join(output_path, 'kernel_info'))
            
            plt.savefig(os.path.join(output_path, 'kernel_info.png') )


    def run_simulation(self) -> None:
        self.animate()
        datetime_dir = str(datetime.now())
        outfile = 'output.gif'   
        print('./folder/{}...)'.format(datetime_dir))
        
        self.save_animation(datetime_dir, outfile)
        self.plot_kernel_info(dir=datetime_dir, save=True)
        return self.lenia_board_state.values()


    def record_board_state(self, i):
        board_arr = self.board.flatten()
        board_val_greater_than_point_five = list(board_arr[board_arr > 0.5])
        self.lenia_board_state["frame_"+str(i)] = len(board_val_greater_than_point_five)


