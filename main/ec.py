import random
import numpy as np
import shutil
import os
from lenia import Lenia
import statistics
from matplotlib import pyplot as plt
import sys




kernel_size = 3
board_size = 64
mutation_rate = 0.1
population_size = 5
generation = 5
gen_best_fitness = {}
gen_average_fitness = {}
each_gen_fitness = []

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

    def calc_fitness(self, board, gen):
        lenia = Lenia(self.genes, board)
        board_alive_cell = lenia.run_simulation("gen_"+str(gen))

        self.fitness = statistics.pstdev(board_alive_cell.values())
        self.plot_output(board_alive_cell, "outputs/gen_"+str(gen))
        return self.fitness
    
    def plot_output(self, board_alive_cell, dir):
        plt.bar(board_alive_cell.keys(), board_alive_cell.values(), width=.5, color='g')
        plt.savefig(dir+'/hist.png')


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
    
    @staticmethod
    def select_roulette_wheel(pop, board, gen):
        original_individuals = pop.individuals
        individual_length = len(original_individuals)
        new_individuals = []
        total_sum = 0
        total_sum = sum([(total_sum + ind.calc_fitness(board, gen)) for ind in pop.individuals])
        random_num = random.randrange(0,int(round(total_sum)))
        partial_sum = 0
        while len(new_individuals) != individual_length - 1:
            for c in original_individuals:
                partial_sum += c.fitness
                if(partial_sum >= random_num):
                    new_individuals.append(c)
                    total_sum = total_sum - c.fitness
                    original_individuals.remove(c)
                    break
        return new_individuals
        

def run_ga(pop_size, generation):
    # sys.stdout = open('logs.txt','wt')
    population = Population(pop_size)
    board = population.board
    for gen in range(1, generation+1):
        gen_fitness_dict = {}
        print("Generation: ", gen, " started")
        elite_individuals = []
        
        population.individuals.sort(key=lambda x: x.calc_fitness(board, gen), reverse= True)
        gen_best_fitness["gen_"+str(gen)] = population.individuals[0].fitness
        all_fitness = [ind.fitness for ind in population.individuals]
        gen_fitness_dict["gen_"+str(gen)] = all_fitness
        each_gen_fitness.append[gen_fitness_dict]
        gen_average_fitness["gen_"+str(gen)] = sum(all_fitness)/len(all_fitness)
        for ind in population.individuals:
            print("--------gene--------")
            print(ind.genes)
            print("--Fitness--: ", ind.fitness)

         # calculate pop fitness and sort it
        elite_individuals.append(population.individuals[0])
        for ind in elite_individuals:
            print("--------Elite gene--------")
            print(ind.genes)
            print("--Elite Fitness--: ", ind.fitness)
        mutated_population = GeneticAlgorithm.mutate_population(population)
        for ind in mutated_population.individuals:
            print("--------mutated gene--------")
            print(ind.genes)
            print("--mutated gene Fitness--: ", ind.fitness)
        
        selected_mutated_individuals = GeneticAlgorithm.select_roulette_wheel(mutated_population, board, gen)
        population.individuals = elite_individuals + selected_mutated_individuals
        print("Generation ",gen, " completed")
        print("---------------------------")
        print("---------------------------")

    # final_pop = population.individuals.sort(key=lambda x: x.calc_fitness(population.board), reverse= True)
    # print("Final population sorted by fitness after generation: ", generation)
    # for indi in final_pop.individuals:
    #     print(final_pop)

    print("gen_best_fitness: ", gen_best_fitness)
    print("gen_average_fitness: ", gen_average_fitness)

if __name__ == "__main__":
    
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
    run_ga(population_size, generation)
  


























