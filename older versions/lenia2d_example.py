from lenia import Lenia

import numpy as np

import matplotlib

matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from matplotlib import animation


world_size = 100
radius = 5
time_step = 0.05
growth_mean = 0.135
growth_std = 0.015
kernel_size = 4 * radius + 1

# Some extra parameters for different behaviour




world = np.random.uniform(size=(world_size, world_size))
kernel = np.ones(shape=(kernel_size, kernel_size))
kernel[radius, radius] = 0
kernel = kernel / np.sum(kernel)

# Use all the previously defined stuff to create a 2 dimensional
# instance of Lenia
lenia2d = Lenia(world, kernel, time_step, growth_mean, growth_std)


def my_func(i):
    # print(i)
    lenia2d.update()
    plt.imshow(
        lenia2d.world,
        vmin=0,
        vmax=1,
    )


fig = plt.figure()
anim = animation.FuncAnimation(
    fig=fig, func=my_func, frames=20, interval=50, blit=False
)

# plt.show()
anim.save("lenia.gif", dpi=300, writer=animation.PillowWriter(fps=25))