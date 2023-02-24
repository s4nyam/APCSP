
# Imports

import numpy as np
from matplotlib import pyplot as plt
import time
import warnings
warnings.simplefilter("ignore", UserWarning)
#Imports
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import scipy.signal
import os.path
import os
from datetime import datetime
from json import JSONEncoder
import json



OUTPUT_PATH = './new_outputs'
MAX_FRAMES = 3000
# zeros ones random sparse gaussian ring

kernel_type = 'circular_kernel'
# square_kernel circular_kernel ring_kernel smooth_ring_kernel kernel_shell

sigma = 0.1
mu = 0.1
dt = 0.1

kernel_size = 16
# kernel = None
# board = None
board_size = 64

frames = 100
seed = None
kernel_peaks = np.array([1])
kernel_diameter = 16
frame_intervals = float(50)


class Lenia:
    def __init__(self):
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.mu = mu
        self.dt = dt
        self.kernel_size = kernel_size
        self.kernel_diameter = kernel_diameter
        self.kernel_peaks = kernel_peaks
        self.kernel = self.initalise_kernel(self.kernel_diameter, peaks=self.kernel_peaks)
        self.normalise_kernel()
        self.board_size = board_size
        self.frames = frames
        self.seed = seed
        
        
        self.frame_intervals = frame_intervals
        self.anim = None
        self.board = np.random.rand(self.board_size, self.board_size)
        self.cmap = 'viridis'
        self.fig, self.img = self.show_board()


    
    def initialise_board(self):
        self.board = np.random.rand(self.grid_size, self.grid_size)
        

    def initalise_kernel(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):

        R = int(diameter / 2) + 1
        D = np.linalg.norm(np.asarray(np.ogrid[-R:R-1, -R:R-1]) + 1) / R
        k = len(peaks)
        kr = k * D
        peak = peaks[np.minimum(np.floor(kr).astype(int), k-1)]
        gaussian = lambda x, m, s: a*np.exp(-( (x-m)**2 / (2*s**2) ))
        self.kernel = (D<1) * gaussian(kr % 1, kernel_mu, kernel_sigma) * peak
        return self.kernel

        

    def growth_function(self, U:np.array):
        gaussian = lambda x, m, s: np.exp(-( (x-m)**2 / (2*s**2) ))
        return gaussian(U, self.mu, self.sigma)*2-1
        

    def show_board(self, 
                   display:bool=False,
                   ):
        """Create figure to display the board. 
        Used to animate each frame during the simulation.

        Args:
            display (bool, optional): Show the figure

        Returns:
            tuple(plt.figure, plt.imshow): Figure and axes items for the board at timestate t.
        """
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

        self.update_convolutional()
        self.img.set_array(self.board) # render the updated state 
        return self.img,
    
    
    def update_convolutional(self) -> np.array:
        """Update the board state using the convolution method to calculate the neighbourhood sum
        
        f(x,y,t+1) = g(k*f(x,y,t))
        
        where         
        f(x,y,t) is the state at time t
        k is the kernel (e.g. Extended Moore neighbourhood)
        g is the growth function
        n.b. the operator '*' represents the convolution operator
        
        Returns:
            np.array: The updated board f(x,y,t+1)
        """
        
        # Calculate the neighbourhood sum by convolution with the kernel.
        # Use periodic boundary conditions to 'wrap' the grid in the x and y dimensions
        neighbours = scipy.signal.convolve2d(self.board, self.kernel, mode='same', boundary='wrap')
        
        # Update the board as per the growth function and timestep dT, clipping values to the range 0..1
        self.board = np.clip(self.board + self.dt * self.growth_function(neighbours), 0, 1)

        return self.board
    
    def save_animation(self, 
                       filename:str,
                       ):
        if not self.anim:
            raise Exception('ERROR: Run animation before attempting to save')
            return 
        
        fmt = os.path.splitext(filename)[1] # isolate the file extension
        
        try: # make outputs folder if not already exists
            os.makedirs(OUTPUT_PATH)
        except FileExistsError:
            # directory already exists
            pass

        if fmt == '.gif':
            f = os.path.join(OUTPUT_PATH, filename) 
            writer = matplotlib.animation.PillowWriter(fps=30) 
            self.anim.save(f, writer=writer)
        else:
            raise Exception('ERROR: Unknown save format. Must be .gif or .mp4')


    
    def normalise_kernel(self) -> np.array:
        """Normalise the kernel such the values sum to 1. 
        This makes generalisations much easier and ensures that the range of the neighbourhood sums is independent 
        of the kernel used. 
        Ensures the values of the growth function are robust to rescaling of the board/kernel. 

        Returns:
            np.array: The resulting normlised kernel
        """
        kernel_norm = self.kernel / (1*np.sum(self.kernel))
        self.norm_factor = 1/ (1*np.sum(self.kernel))
        self.kernel = kernel_norm 
        return kernel_norm
        
        
    def plot_kernel_info(self,
                         cmap:str='viridis', 
                         bar:bool=False,
                         save:str=None,
                         ) -> None:
        """Display the kernel, kernel cross-section, and growth function as a matplotlib figure.

        Args:
            kernel (np.array): The kernel to plot
            growth_fn (object): The growth function used to update the board state
            cmap (str, optional): The colourmap to use for plotting. Defaults to 'viridis'.
                                (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
            bar (bool, optional): Plot the kernel x-section as a bar or line plot. Defaults to False.
            save (str, optional): Save the figure
        """
        
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
        ax[2].plot(x, self.growth_function(x))
        
        if save:
            print('Saving kernel and growth function info to', os.path.join(OUTPUT_PATH, 'kernel_info'))
            print(str(datetime.now()))
            plt.savefig(os.path.join(OUTPUT_PATH, str(datetime.now())+"_"+'kernel_info_'+self.kernel_type+'.png') )


    def run_simulation(self) -> None:
 
       
        self.animate()
        outfile = 'output.gif'   
        print('./new_outputs/{}...)'.format(outfile))
        self.save_animation(outfile)
        # self.plot_kernel_info(save=True)


if __name__ == "__main__":
    
    lenia = Lenia()
    lenia.run_simulation()