import random
import numpy as np

from lenia import Lenia



    # INTERESTING KERNEL
def spider_web_kernel():
        m=10
        n=10
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
        # smoothing_factor = 0.5
        smoothing_factor = random.uniform(0,1)
        
        # calculate the values for each point
        grid = np.sin(distance * smoothing_factor) * distance
        
        return grid

def random_kernel_generator():
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
        

    return grid
# image = grid
# # Display the resulting image
# import matplotlib.pyplot as plt
# plt.imshow(image, cmap='gist_earth')
# plt.show()



mutation_rate = 0.1

class Individual:

    def __init__(self):
        self._gene = spider_web_kernel()
        self._fitness = 0

    def calc_fitness(self):
        lenia = Lenia(self._gene)
        board_alive_cell = lenia.run_simulation()
        return sum(board_alive_cell)
    
    def get_genes(self):
        return self._gene

class Population:
    def __init__(self, size):
        self._individuals = []
        for _ in range(0, size):
            self._individuals.append(Individual())

    # def set_indi

    def get_individuals(self):
        return self._individuals

class GeneticAlgorithm:


    @staticmethod
    def mutate_population(population):
        for i in range(len(population.get_individuals())):
            population.get_individuals()[i] = GeneticAlgorithm._mutate_individual(population.get_individuals()[i])
        return population

    @staticmethod
    def _mutate_individual(individual_kernel):
        print("Mutation function")
        if random.random() < mutation_rate:
            print("Kernel got mutated")
            random_no_for_mutation = 20
            random_indices = np.random.choice(individual_kernel.get_genes().size, random_no_for_mutation, replace=False )
            individual_kernel.get_genes().flat[random_indices] = np.random.rand(5)
        return individual_kernel

        

def run_ga(pop_size, generation):
    population = Population(pop_size)
    for gen in range(1, generation+1):
        print("Generation: ", gen)
        elite = []
        # calculate pop fitness and sort it
        population.get_individuals().sort(key=lambda x: x.calc_fitness(), reverse= True)
        # print("population len : ",len(population))
        # elite.append(population.get_individuals()[0])
        # mutated_population = GeneticAlgorithm.mutate_population(population)
        # new_individuals = elite + mutated_population.get_individuals()[0:pop_size -1]
        population._individuals = population.get_individuals()
        print(len(population._individuals))
        print("Generation ",gen, " completed")
        print("---------------------------")
        print("---------------------------")


if __name__ == "__main__":
    run_ga(3, 2)
  



























