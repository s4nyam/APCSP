import random
from cellular_automata.Lenia import Lenia

class Individual:
  def __init__(self):
    self._genes = {
                    'board': None, 
                    'board_size': 64, 
                    'kernel_size': 16, 
                    'kernel_peaks': None, 
                    'mu': round(random.uniform(0, 1), 2), 
                    'sigma': round(random.uniform(0, 1), 2), 
                    'dt': 0.1, 
                    'frames': 100, 
                    'seed': None
                  }
    self._fitness = 0


    def get_fitness(self):
      lenia= Lenia(self._genes)
      live_cell_counts = lenia.run_simulation()
      self._fitness = live_cell_counts["live_cells"]
      return self._fitness
  
  def __str__(self):
    return self._genes.__str__()