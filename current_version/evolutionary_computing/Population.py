from evolutionary_computing.Individual import Individual


class Population:
  def __init__(self, size):
    self._individuals = []
    for _ in range(0, size):
      self._individuals.append(Individual())
  
  def get_individuals(self):
    return self._individuals
