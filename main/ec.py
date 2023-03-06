import random
import numpy as np
import shutil
import os
from lenia import Lenia


kernel_size = 16
board_size = 64
mutation_rate = 0.01
population_size = 20
generation = 100

def random_kernel_generator():
    grid = np.random.rand(kernel_size, kernel_size)
    grid = np.round(grid, 3)
    return grid

def random_board_generator():
    board = np.random.rand(board_size, board_size)
    board = np.round(board, 3)
    return board

class Individual:

    def __init__(self):
        self.genes = random_kernel_generator()
        self.fitness = 0

    def calc_fitness(self, board):
        lenia = Lenia(self.genes, board)
        board_alive_cell = lenia.run_simulation()
        self.fitness = sum(board_alive_cell)
        return self.fitness

class Population:
    def __init__(self, size):
        self.individuals = []
        self.board = random_board_generator()
        for _ in range(0, size):
            self.individuals.append(Individual())


class GeneticAlgorithm:

    @staticmethod
    def mutate_population(population):
        for i in range(len(population.individuals)):
            population.individuals[i] = GeneticAlgorithm._mutate_individual(population.individuals[i])
        return population

    @staticmethod
    def _mutate_individual(individual):
        kernel = individual.genes
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                if random.random() < mutation_rate:
                    kernel[i][j] = np.round(np.random.rand(), 3)
        individual.genes = kernel
        return individual
        

def run_ga(pop_size, generation):
    population = Population(pop_size)

    for gen in range(1, generation+1):
        print("Generation: ", gen, " started")
        elite_individuals = []
        no_selected_mutated_ind = pop_size - 1
       
        population.individuals.sort(key=lambda x: x.calc_fitness(population.board), reverse= True)
         # calculate pop fitness and sort it
        elite_individuals.append(population.individuals[0])
        mutated_population = GeneticAlgorithm.mutate_population(population)
        mutated_individuals = random.sample(mutated_population.individuals, no_selected_mutated_ind)
        population.individuals = elite_individuals + mutated_individuals
        print("Generation ",gen, " completed")
        print("---------------------------")
        print("---------------------------")

    final_pop = population.individuals.sort(key=lambda x: x.calc_fitness(population.board), reverse= True)
    print("Final population sorted by fitness after generation: ", generation)
    for indi in final_pop.individuals:
        print(final_pop)

if __name__ == "__main__":
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
    run_ga(population_size, generation)
  



























