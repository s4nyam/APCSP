import random
import numpy as np
import shutil
import os
# from lenia import Lenia
import statistics
from matplotlib import pyplot as plt
import sys
import copy


kernel_size = 16
board_size = 64
mutation_rate = 0.1
population_size = 3
generation = 10
gen_best_fitness = {}
gen_average_fitness = {}
each_gen_fitness = []
no_of_elites = 1

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
    return self.fitness
    

class Population:
  def __init__(self, size):
    self.individuals = []
    self.board = random_board_generator()
    for _ in range(0, size):
        self.individuals.append(Individual())


class GeneticAlgorithm:

  @staticmethod
  def mutate_individuals(individuals):
    mutated_individuals = []
    for ind in individuals:
        mutated_ind = GeneticAlgorithm._mutate_individual(ind)
        mutated_individuals.append(mutated_ind)
    return mutated_individuals

  @staticmethod
  def _mutate_individual(ind):
    kernel = ind.genes
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if random.random() < mutation_rate:
                kernel[i][j] = np.round(np.random.rand(), 3)
    ind.genes = kernel
    return ind
    
  @staticmethod
  def select_roulette_wheel(individuals, board, gen):
    original_individuals = individuals
    individual_length = len(original_individuals)
    new_individuals = []
    total_sum = 0
    total_sum = sum([(total_sum + ind.calc_fitness(board, gen)) for ind in individuals])
    print("Mutated population fitness: ",  [ind.fitness for ind in individuals])
    random_num = random.randrange(0,int(round(total_sum)))
    partial_sum = 0
    while len(new_individuals) != individual_length - no_of_elites:
        for c in original_individuals:
            partial_sum += c.fitness
            if(partial_sum >= random_num):
                new_individuals.append(c)
                total_sum = total_sum - c.fitness
                original_individuals.remove(c)
                break
    
    print("Roulette - Mutated population fitness: ",  [ind.fitness for ind in new_individuals])
    return new_individuals
        

def run_ga(pop_size, generation):
  # sys.stdout = open('logs.txt','wt')
  population = Population(pop_size)
  board = population.board
  for gen in range(1, generation+1):
      print("Generation: ", gen, " started")
      gen_fitness_dict = {}
      population.individuals.sort(key=lambda x: x.calc_fitness(board, gen), reverse= True)
      gen_best_fitness["gen_"+str(gen)] = population.individuals[0].fitness
      all_fitness = [ind.fitness for ind in population.individuals]
      gen_fitness_dict["gen_"+str(gen)] = all_fitness
      print("Fitness of this generation: ", all_fitness)
      each_gen_fitness.append(gen_fitness_dict)
      gen_average_fitness["gen_"+str(gen)] = sum(all_fitness)/len(all_fitness)
      elite_individuals = [copy.deepcopy(population.individuals[i]) for i in range(0,no_of_elites)]
      mutated_individuals = GeneticAlgorithm.mutate_individuals(population.individuals)
      selected_mutated_individuals = GeneticAlgorithm.select_roulette_wheel(mutated_individuals, board, gen)
      population.individuals = elite_individuals + selected_mutated_individuals
      print("elite fitness: ", [ind.fitness for ind in elite_individuals])
      print("selected mutated fitness: ", [ind.fitness for ind in selected_mutated_individuals])
      print("Next generation fitness : ", [ind.fitness for ind in population.individuals])
      print("Generation ",gen, " completed")
      print("---------------------------")
      print("---------------------------")

  print("gen_best_fitness: ", gen_best_fitness)
  plot_figures(gen_best_fitness, "gen_best_fitness.png")
  print("gen_average_fitness: ", gen_average_fitness)
  plot_figures(gen_average_fitness, "gen_average_fitness.png")


def plot_figures(data, name):
  plt.clf()
  labels = list(data.keys())
  values = list(data.values())
  plt.plot(labels, values)
  plt.xlabel('Generation')
  plt.ylabel('Fitness')
  plt.title('Fitness of Generations')
  plt.savefig(name) 

if __name__ == "__main__":
    
  if os.path.exists('outputs'):
    shutil.rmtree('outputs')
  run_ga(population_size, generation)
