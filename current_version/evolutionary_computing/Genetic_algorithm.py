class GeneticAlgorithm:
  @staticmethod
  def evolve(population, selection_criteria, crossover_type, mutation_rate):
    # crossover_pop = GeneticAlgorithm._crossover_population(population, selection_criteria, crossover_type)

    return GeneticAlgorithm._mutate_population(population, mutation_rate)
  

  # mutating and returning the population
  @staticmethod
  def _mutate_population(pop, mutation_rate):
    for i in range(elite_individuals):
      GeneticAlgorithm._mutate_individual(pop.get_individuals()[i], mutation_rate)
    return pop



# mutation_rate = 0.25
# the rate should be less in value, this is for the test purpose
# mutation is a background operator whose is used to prevent the algorithm from 
# prematurely converging to a suboptimal solution

# mutating the individual
  @staticmethod
  def _mutate_individual(individual, mutation_rate):
    for i in range(target_individual.__len__()):

      if random.random() < mutation_rate:
        if random.random() < 0.5:
          individual.get_genes()[i] = 1
        else:
          individual.get_genes()[i] = 0

  
# tournament_selection_size = 4
# return randomly select 4 individuals from the population after 
# sorting them by fitness
  @staticmethod
  def _select_tournament_population(pop):
    # empty population
    tournament_pop = Population(0)
    i = 0

    # randomly select four sets of individual
    while i < tournament_selection_size:
      tournament_pop.get_individuals().append(pop.get_individuals()[random.randrange(0, population_size)])
      i += 1
    
    # sort by descending order 
    tournament_pop.get_individuals().sort(key=lambda x: x.get_fitness(), reverse = True)
    return tournament_pop


  @staticmethod
  def _select_roulette_wheel(pop):
    population = Population(0)
    roulette_wheel_result = -1
    total_sum = 0
    total_sum = sum([(total_sum + c.get_fitness()) for c in pop.get_individuals()])
    random_num = random.randrange(0,total_sum)
    partial_sum = 0
    for c in pop.get_individuals():
      partial_sum += c.get_fitness()
      if(partial_sum >= random_num):
        population.get_individuals().append(c)
        break;
    return population;
iterations = 30
def multiple_run_ga(selection_criteria, crossover_type= "SINGLE_POINT", individual_numbers = population_size, mutation_rate = mutation_rate_val):
  each_run_average = []
  each_run_stdev = []
  each_run_generation = []


  for i in range(iterations):
    # print("ITERATION NUMBER",str(i))
    generations_average_for_a_run = [] 
    generations_stdev_for_a_run = []
    population = Population(individual_numbers)
    population.get_individuals().sort(key=lambda x: x.get_fitness(), reverse= True)
    stats = _print_population(population, 0)
    generations_average_for_a_run.append(stats[0])
    generations_stdev_for_a_run.append(stats[1])
    generation_number = 1
    

    while population.get_individuals()[0].get_fitness() < target_individual.__len__():
      population = GeneticAlgorithm.evolve(population, selection_criteria, crossover_type, mutation_rate)
      population.get_individuals().sort(key=lambda x: x.get_fitness(), reverse= True)
      logging.info("ITERATION NUMBER : "+ str(i))
      stats = _print_population(population, generation_number)
      generations_average_for_a_run.append(stats[0])
      generations_stdev_for_a_run.append(stats[1])
      generation_number += 1

    logging.info("generations_average_for_a_run : "+ str(generations_average_for_a_run) )
    each_run_average.append(mean(generations_average_for_a_run))
    logging.info("STANDARD DEVIATION : "+ str(each_run_stdev))
    if len(generations_stdev_for_a_run) > 1:
      each_run_stdev.append(stdev(generations_stdev_for_a_run))
    else:
      each_run_stdev.append(generations_stdev_for_a_run[0])
    each_run_generation.append(generation_number)
  logging.info("Average of all run : ")
  logging.info(each_run_average)
  logging.info("standard deviation of all run : ")
  logging.info(each_run_stdev)
  logging.info("GENERATION OF ALL RUN : ")
  logging.info(each_run_generation)
  logging.info("-----------------------------------------------------")
  return (each_run_average, each_run_stdev, each_run_generation)