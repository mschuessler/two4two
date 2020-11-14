import matplotlib.pyplot as plt
import numpy as np

class ColorGenerator():

    def __init__(self, palette=None):
        self.cmap = plt.get_cmap(palette)

    def get_color(self, val=None):
        if val is not None:
            return self.cmap(val)
        elif type(self.cmap) == list:
            cm = np.random.choice(self.cmap)
            return cm(np.random.uniform())
        else:
            return self.cmap(np.random.uniform())

    def get_random_color(self):
        color = np.random.uniform(size=3)
        return tuple(color) + (1,)