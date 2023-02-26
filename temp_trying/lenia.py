import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Lenia:
    def __init__(self, size, kernel, growth_function, automaton):
        self.size = size
        self.kernel = kernel
        self.growth_function = growth_function
        self.automaton = automaton
        self.board = np.zeros((size, size))

    def update(self):
        next_board = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                neighbors = self.kernel(self.board, i, j)
                next_board[i, j] = self.automaton(neighbors, self.growth_function(self.board[i, j]))
        self.board = next_board

    def generate(self, frames):
        fig, ax = plt.subplots()
        plt.gray()
        plt.axis('off')
        img = ax.imshow(self.board, animated=True)

        def animate(frame):
            self.update()
            img.set_data(self.board)
            return img

        anim = FuncAnimation(fig, animate, frames=frames, repeat=False)
        return anim

    def save_as_gif(self, filename, frames):
        anim = self.generate(frames)
        anim.save(filename, writer='imagemagick', fps=30)

def kernel(board, i, j):
    x, y = np.indices((board.shape[0], board.shape[1]))
    r = np.sqrt((x-i)**2 + (y-j)**2)
    return board[r <= 2.0]

def growth_function(x):
    return 1.0 - x

def automaton(neighbors, current):
    n = np.sum(neighbors) - current
    if current == 0:
        if n == 3:
            return 1
        else:
            return 0
    else:
        if n < 2 or n > 3:
            return 0
        else:
            return 1

lenia = Lenia(256, kernel, growth_function, automaton)
lenia.save_as_gif('lenia.gif', frames=40)
