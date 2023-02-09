from scipy import signal
import numpy as np


class Lenia:
    def __init__(self, world_init, kernel, time_step, growth_mean, growth_std):
        self.world = world_init
        self.kernel = kernel
        self.time_step = time_step

        self.growth_mean = growth_mean
        self.growth_std = growth_std

    def update(self):
        convolved_world = signal.convolve(self.world, self.kernel, mode="same")
        self.world = self._clip(
            self.world + self.time_step * self._growth(convolved_world)
        )

    def _growth(self, x):
        return 2 * np.exp(-(((x - self.growth_mean) / self.growth_std) ** 2) / 2) - 1

    def _clip(self, x):
        return np.clip(x, 0, 1)
