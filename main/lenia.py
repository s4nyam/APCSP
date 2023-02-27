
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
import itertools




kernel_list = [
#                  "circle_smooth",
#                  "concentric_circle_non_smootheened_kernel",
#                  "solid_circle_kernel",
#                  "glider_kernel",
#                  "glider_gun_kernel",
#                  "fixed_central_initialisation_kernel",
#                  "rps_glider2",
#                  "concentric_sqaure_circle",
#                  "meshgrid_concentric_circle",
#                  "corner_circle_kernel",
#                  "mesh_circle_kernel",
#                  "labyrinth_kernel",
                  "spider_web_kernel",
                    # "concentric_circle_smooth"
                 ]
growth_fn_list = [
                    "growth_function1",
                    # "growth_function2",
                    # "growth_function3",
                    # "growth_function4",
                    # "growth_function5"
                ]

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

class Lenia:
    def __init__(self, kernel_type, growthfn_type):
        self.sigma = sigma
        self.mu = mu
        self.dt = dt
        self.kernel_size = kernel_size
        self.kernel_diameter = kernel_diameter
        self.kernel_peaks = kernel_peaks
        self.kernel = self.choose_kernel(kernel_type)
        self.normalise_kernel()
        self.board_size = board_size
        self.frames = frames
        self.seed = seed
        self.growth = self.choose_growth_function(growthfn_type)
        
        self.frame_intervals = frame_intervals
        self.anim = None
        
        
        # For random initialisation
        # self.board = np.random.rand(self.board_size, self.board_size)
        
        # For single pixel in center initialisation
        # self.board = np.zeros((self.board_size, self.board_size))
        
        # For probabilistic initialisation
        prob = 0.05
        self.board = np.random.choice([0, 1], size=(self.board_size, self.board_size), p=[1-prob, prob])

        self.board[self.board_size//2, self.board_size//2] = 1
        self.cmap = 'viridis'
        self.fig, self.img = self.show_board()
        


    def choose_growth_function(self, growthfn_variation):
        if growthfn_variation == "growth_function1":
            growth_fn = self.growth_function1
        elif growthfn_variation == "growth_function2":
            growth_fn = self.growth_function2
        elif growthfn_variation == "growth_function3":
            growth_fn = self.growth_function3
        elif growthfn_variation == "growth_function4":
            growth_fn = self.growth_function4
        elif growthfn_variation == "growth_function5":
            growth_fn = self.growth_function5
        return growth_fn
    

    def choose_kernel(self, kernel_variation):
        if kernel_variation == "circle_smooth":
            kernel = self.circle_smooth(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "concentric_circle_non_smootheened_kernel":
            kernel = self.concentric_circle_non_smootheened_kernel(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "solid_circle_kernel":
            kernel = self.solid_circle_kernel(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "glider_kernel":
            kernel = self.glider_kernel(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "glider_gun_kernel":
            kernel = self.glider_gun_kernel(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "fixed_central_initialisation_kernel":
            kernel = self.fixed_central_initialisation_kernel(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "rps_glider1":
            kernel = self.rps_glider1(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "rps_glider2":
            kernel = self.rps_glider2(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "concentric_sqaure_circle":
            kernel = self.concentric_sqaure_circle(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "meshgrid_concentric_circle":
            kernel = self.meshgrid_concentric_circle(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "corner_circle_kernel":
            kernel = self.corner_circle_kernel(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "mesh_circle_kernel":
            kernel = self.mesh_circle_kernel(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "labyrinth_kernel":
            kernel = self.labyrinth_kernel(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "spider_web_kernel":
            kernel = self.spider_web_kernel(self.kernel_diameter, peaks=self.kernel_peaks)
        elif kernel_variation == "concentric_circle_smooth":
            kernel = self.concentric_circle_smooth(self.kernel_diameter, peaks=self.kernel_peaks)
        return kernel

    # KERNELS AND ITS TWEAKS - KERNELS BEGIN HERE
    # KERNELS AND ITS TWEAKS - KERNELS BEGIN HERE
    # KERNELS AND ITS TWEAKS - KERNELS BEGIN HERE
    # KERNELS AND ITS TWEAKS - KERNELS BEGIN HERE


    # FLEXIBLITY TO CHANGE KERNEL
    def circle_smooth(self, 
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
        # print(self.kernel.shape)
        return self.kernel
        

    # # 24 feb 9 AM - concentric circle 
    # # 24 feb 9 AM - concentric circle 
    # # 24 feb 9 AM - concentric circle 
    # # 24 feb 9 AM - concentric circle 
        # [[3, 3, 3, 3, 3, 3, 3],
        # [3, 2, 2, 2, 2, 2, 3],
        # [3, 2, 1, 1, 1, 2, 3],
        # [3, 2, 1, 0, 1, 2, 3],
        # [3, 2, 1, 1, 1, 2, 3],
        # [3, 2, 2, 2, 2, 2, 3],
        # [3, 3, 3, 3, 3, 3, 3]]

    def concentric_circle_non_smootheened_kernel(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        diameter = 17
        if diameter < 1:
            return []

        radius = diameter // 2
        circle = [[0 for x in range(diameter)] for y in range(diameter)]

        for y in range(diameter):
            for x in range(diameter):
                distance = ((x - radius) ** 2 + (y - radius) ** 2) ** 0.5
                if distance <= radius:
                    circle[y][x] = radius - int(distance)

        return circle


    # 24 feb 9:20 AM - Solid Circle
    # 24 feb 9:20 AM - Solid Circle
    # 24 feb 9:20 AM - Solid Circle
    # 24 feb 9:20 AM - Solid Circle

    # [[0, 0, 1, 1, 1, 0, 0],
    #  [0, 1, 1, 1, 1, 1, 0],
    #  [1, 1, 1, 1, 1, 1, 1],
    #  [1, 1, 1, 1, 1, 1, 1],
    #  [1, 1, 1, 1, 1, 1, 1],
    #  [0, 1, 1, 1, 1, 1, 0],
    #  [0, 0, 1, 1, 1, 0, 0]]

    def solid_circle_kernel(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        if diameter < 1:
            return []
        diameter = 17
        radius = diameter // 2
        circle = [[0 for x in range(diameter)] for y in range(diameter)]

        for y in range(diameter):
            for x in range(diameter):
                distance = ((x - radius) ** 2 + (y - radius) ** 2) ** 0.5
                if distance <= radius:
                    circle[y][x] = diameter

        return circle

    # 24 feb 9:29 AM - Glider
    # 24 feb 9:29 AM - Glider
    # 24 feb 9:29 AM - Glider
    # 24 feb 9:29 AM - Glider

    # [[0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 0, 0, 0, 0],
    # [0, 0, 0, 1, 0, 0, 0],
    # [0, 1, 1, 1, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0]]

    def glider_kernel(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        diameter = 13
        if diameter < 3:
            return []

        # Initialize the grid with all cells set to 0
        grid = [[0 for x in range(diameter)] for y in range(diameter)]

        # Set the cells for the glider pattern
        grid[1][2] = 1
        grid[2][3] = 1
        grid[3][1] = 1
        grid[3][2] = 1
        grid[3][3] = 1

        return grid
    
    
    # 24 feb 9:29 AM - Glider GUN
    # 24 feb 9:29 AM - Glider GUN
    # 24 feb 9:29 AM - Glider GUN
    # 24 feb 9:29 AM - Glider GUN
    # 24 feb 9:29 AM - Glider GUN
    def glider_gun_kernel(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        m = 50
        n = 50
        pop = np.zeros([m, n], dtype = int)
        pop[6, 2] = 1
        pop[7, 2] = 2
        pop[6, 3] = 2
        pop[7, 3] = 1
        pop[6:9, 12] = 2
        pop[5, 13] = 2
        pop[9, 13] = 1
        pop[4, 14:16] = 2
        pop[10, 14:16] = 2
        pop[7, 16] = 1
        pop[5, 17] = 1
        pop[9, 17] = 2
        pop[6, 18] = 1
        pop[8, 18] = 2
        pop[7, 18:20] = 1
        pop[4:7, 22:24] = 1
        pop[3, 24] = 1
        pop[7, 24] = 2
        pop[2:4, 26] = 1
        pop[7:9, 26] = 2
        pop[4, 36] = 1
        pop[5, 36] = 2
        pop[4, 37] = 2
        pop[5, 37] = 1
        return pop


    # 24 feb 9:29 AM - Fixed central initialisation
    # 24 feb 9:29 AM - Fixed central initialisation
    # 24 feb 9:29 AM - Fixed central initialisation
    # 24 feb 9:29 AM - Fixed central initialisation

    def fixed_central_initialisation_kernel(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        m = 50
        n = 50
        pop = np.zeros([m, n], dtype = int)
        midrow = m // 2
        midcol = n // 2
        pop[midrow - 3, midcol - 1] = 3
        pop[midrow - 3, midcol + 1] = 3
        pop[midrow - 1, (midcol - 1):(midcol + 2)] = 3
        pop[midrow + 1, (midcol - 1):(midcol + 2)] = 3
        pop[midrow + 3, midcol:(midcol + 2)] = 3
        return pop

    def rps_glider1(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        dim = [50,50]
        fixed1 = 1
        fixed2 = 2
        seed = 10
        np.random.seed(seed)
        pop = np.zeros((dim[0],dim[1]))
        pop[0, :] = fixed1
        pop[dim[0] - 1, :] = fixed1
        pop[:, 0] = fixed1
        pop[:, dim[1] - 1] = fixed1
        
        pop[1, :] = fixed2
        pop[dim[0] - 2, :] = fixed2
        pop[:, 1] = fixed2
        pop[:, dim[1] - 2] = fixed2

        return pop
        

    def rps_glider2(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        m = 50
        n = 50
        pop = np.zeros([m, n], dtype = int)
        midrow = m // 2 - 5
        midcol = n // 2 - 5
        pop[midrow:(midrow + 10), midcol:(midcol + 10)] = rps_glider1(10, 10)
        return pop



    def concentric_sqaure_circle(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        diameter = 17
        if diameter < 1:
            return []

        # Initialize the grid with all cells set to 0
        grid = [[0 for x in range(diameter)] for y in range(diameter)]

        # Set the cells for the outermost square
        for i in range(diameter):
            grid[0][i] = diameter
            grid[i][0] = diameter
            grid[diameter-1][i] = diameter
            grid[i][diameter-1] = diameter

        # Set the cells for the inner squares
        for i in range(2, diameter//2+1, 2):
            for j in range(i-1, diameter-i+1):
                grid[i-1][j] = diameter-j
                grid[j][i-1] = diameter-j
                grid[diameter-i][j] = diameter-i
                grid[j][diameter-i] = diameter-i

        return grid

    def meshgrid_concentric_circle(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        import numpy as np
        from scipy.signal import convolve2d
        np.set_printoptions(threshold=np.inf, suppress=True)

        # Define parameters
        width = 50
        height = 50
        center_x = width // 2
        center_y = height // 2
        rings = 20
        blur = 10

        # Create a meshgrid to represent the image
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Calculate distance of each pixel from the center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Create a concentric circle pattern
        circle = np.zeros((width, height))
        for r in range(rings):
            circle[dist < r] = r

        # Apply blur to the circle pattern
        kernel = np.ones((blur, blur)) / blur**2
        circle = convolve2d(circle, kernel, mode='same')

        # Normalize the values to [0, 1]
        circle = (circle - circle.min()) / (circle.max() - circle.min())

        # # Display the resulting image
        # import matplotlib.pyplot as plt
        # plt.imshow(circle, cmap='gist_earth')
        # plt.show()
        # print(circle)

        return circle

    def corner_circle_kernel(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        
        import numpy as np

        # Define the size of the array
        size = 18

        # Create a 2D array of zeros
        arr = np.zeros((size, size))

        # Define the radius and value of the circle
        radius = 4
        value = 1.0

        # Loop through each element in the array
        for i in range(size):
            for j in range(size):
                
                # Calculate the distance from the current element to the center of the array
                x = abs(i - size/2)
                y = abs(j - size/2)
                distance = np.sqrt(x**2 + y**2)
                
                # If the distance is less than or equal to the radius, set the value to the circle value
                if distance <= radius:
                    arr[i,j] = value

        # Define the values and size of the corners
        corner_value = 0.5
        corner_size = 3

        # Set the corner values
        arr[0:corner_size, 0:corner_size] = corner_value
        arr[0:corner_size, size-corner_size:size] = corner_value
        arr[size-corner_size:size, 0:corner_size] = corner_value
        arr[size-corner_size:size, size-corner_size:size] = corner_value

        # Print the array
        

        return arr

    def mesh_circle_kernel(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        
        import numpy as np
        import matplotlib.pyplot as plt

        # Define the size of the grid
        n = 100

        # Define the center of the circle
        cx = n//2
        cy = n//2

        # Define the radius of the circle
        r = n//5

        # Create a meshgrid of indices
        x, y = np.meshgrid(np.arange(n), np.arange(n))

        # Calculate the distance of each point from the center of the circle
        d = np.sqrt((x-cx)**2 + (y-cy)**2)

        # Create a 2D array with the same shape as the meshgrid
        arr = np.zeros((n, n))

        # Set the values of the array within the circle to 1
        arr[d < r] = 1

        # Create a Gaussian kernel for smoothing
        k = np.exp(-d**2/(2*(r/2)**2))

        # Apply the kernel to the array
        arr = k*arr

        # Normalize the array to values between 0 and 1
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # Set the values in the four corners to different values
        arr[:n//4, :n//4] = 0.8
        arr[:n//4, -n//4:] = 0.8
        arr[-n//4:, :n//4] = 0.8
        arr[-n//4:, -n//4:] = 0.8

        # # Plot the array as an image
        # plt.imshow(arr, cmap='gist_earth')
        # plt.show()

        

        return arr


    def labyrinth_kernel(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):

        import numpy as np
        from scipy.ndimage.filters import gaussian_filter

        # Define the size of the labyrinth
        n_rows, n_cols = 50, 50

        # Create a labyrinth with random walls
        labyrinth = np.random.choice([0, 1], size=(n_rows, n_cols), p=[0.6, 0.4])

        # Smooth the labyrinth using the gaussian filter
        sigma = 2  # Controls the amount of smoothing
        smoothed_labyrinth = gaussian_filter(labyrinth.astype(float), sigma=sigma)

        # Normalize the smoothed labyrinth to have values between 0 and 1
        smoothed_labyrinth /= np.max(smoothed_labyrinth)

        # Plot the array as an image
        # plt.imshow(smoothed_labyrinth, cmap='gist_earth')
        # plt.show()
        # Print the resulting labyrinth
        # print(smoothed_labyrinth)

        
        return smoothed_labyrinth


    # INTERESTING KERNEL
    def spider_web_kernel(self, 
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

    # INTERESTING KERNEL
    def concentric_circle_smooth(self, 
                         diameter:int, 
                     peaks:np.array(float)=np.array([1/2, 2/3, 1]), 
                     kernel_mu:float=0.5, 
                     kernel_sigma:float=0.15, 
                     a:float=4.0):
        diameter=20
        radius = diameter // 2
        x, y = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
        distance = np.sqrt(x**2 + y**2)
        mask = np.logical_and(distance <= radius, distance > radius-2)
        smoothed_mask = np.zeros_like(mask, dtype=np.float)
        smoothed_mask[mask] = (distance[mask] - (radius-2)) / 2
        circle = np.zeros_like(x, dtype=np.float)
        circle[mask] = 1
        smoothed_circle = np.zeros_like(x, dtype=np.float)
        smoothed_circle[mask] = circle[mask] * smoothed_mask[mask]
        return smoothed_circle















    # KERNELS AND ITS TWEAKS - KERNELS END HERE
    # KERNELS AND ITS TWEAKS - KERNELS END HERE
    # KERNELS AND ITS TWEAKS - KERNELS END HERE
    # KERNELS AND ITS TWEAKS - KERNELS END HERE





    # FLEXIBLITY TO CHANGE GROWTH FUNCTION
    def growth_function1(self, U:np.array):
        gaussian = lambda x, m, s: np.exp(-( (x-m)**2 / (2*s**2) ))
        return gaussian(U, self.mu, self.sigma)*2-1

    # Dies very fast
    def growth_function2(self, U:np.array):
        k=0.5
        return np.sin(U)*np.cos(U*k)
        # This growth function takes in a numpy array U and returns a numpy array with the same shape as U. 
        # The function applies a sine and cosine transformation to U, with the parameter self.k 
        # controlling the frequency of the cosine component. 
        # This growth function would produce a pattern of cells with oscillating values that change smoothly over time. 
        # It could be used to simulate biological systems that exhibit rhythmic behaviors, 
        # such as circadian rhythms or heartbeat patterns.
    def growth_function3(self, U:np.array):
        k = np.random.uniform(low=0.1, high=0.5, size=U.shape)
        g1 = np.exp(-((U-0.2)**2)/k**2)
        g2 = np.exp(-((U-0.8)**2)/k**2)
        phi = 2
        return (g1+g2)*np.log10(phi*U)        
    

    def growth_function4(self, U:np.array):
        log = lambda x, a: np.log(x+a)
        return log(U, 1.0)
    
    
    def growth_function6(self, U:np.array):
        k=0.5
        return np.sin(U)*(1/np.log(U*k))

    def growth_function5(self, U: np.array):
        exponent = lambda x, a, b, c, d: np.exp(-np.power(a * x - b, 2)) * np.cos(c * x + d)
        return exponent(U, 0.08, 0.2, 10, 0.9)


   




   











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
        self.board = np.clip(self.board + self.dt * self.growth(neighbours), 0, 1)
        self.img.set_array(self.board) # render the updated state 
        return self.img,
    
    
    def update_convolutional(self) -> np.array:
        
        # Calculate the neighbourhood sum by convolution with the kernel.
        # Use periodic boundary conditions to 'wrap' the grid in the x and y dimensions
        neighbours = scipy.signal.convolve2d(self.board, self.kernel, mode='same', boundary='wrap')
        
        # Update the board as per the growth function and timestep dT, clipping values to the range 0..1
        self.board = np.clip(self.board + self.dt * self.growth(neighbours), 0, 1)


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
        ax[2].plot(x, self.growth(x))
        
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



if __name__ == "__main__":  
    kernel_growthfn_combination = list(itertools.product(kernel_list, growth_fn_list, repeat=1))            
    for kernel_growthfn in kernel_growthfn_combination:
        print(kernel_growthfn[0])
        print(kernel_growthfn[1])
        lenia = Lenia(kernel_growthfn[0], kernel_growthfn[1])
        lenia.run_simulation()
