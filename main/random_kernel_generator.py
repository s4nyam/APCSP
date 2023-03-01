import numpy as np

# define the grid size
grid_size = 10

# create an empty grid
grid = np.zeros((grid_size, grid_size))

# set the number of circles to create
num_circles = 5

# loop through and create circles
for i in range(num_circles):
    # set a random diameter between 1 and half the grid size
    diameter = np.random.randint(1, grid_size // 2)
    
    # set a random center point
    x_center = np.random.randint(diameter, grid_size - diameter)
    y_center = np.random.randint(diameter, grid_size - diameter)
    
    # create a mesh grid
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    
    # calculate the distance of each point from the center
    distance = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    
    # set all points within the diameter to 1
    grid[distance < diameter] = np.random.uniform(0,1)
    


image = grid
# Display the resulting image
import matplotlib.pyplot as plt
plt.imshow(image, cmap='gist_earth')
plt.show()