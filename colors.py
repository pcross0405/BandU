import numpy as np
from matplotlib.colors import ListedColormap

class Colors():
    def __init__(self):
        self.reds = self.Colormap([np.array([125, 125, 125])/255, 
                                   np.array([255, 46, 31])/255
        ])
        self.stoplight = self.Colormap([np.array([255, 46, 31])/255, 
                                        np.array([255, 180, 31])/255, 
                                        np.array([26, 209, 23])/255
        ])
        self.GrOrRd = self.Colormap([np.array([145, 145, 145])/255, 
                                     np.array([255, 143, 31])/255, 
                                     np.array([255, 46, 31])/255
        ])
    def Colormap(self, colorlist:list)->ListedColormap:
        num = 250*(len(colorlist)-1)
        parts = 250
        colors = np.empty((num, 4))
        reds = np.zeros(num)
        greens = np.zeros(num)
        blues = np.zeros(num)
        for i, color in enumerate(colorlist):
            if i == len(colorlist) - 1:
                break
            reds[i*parts:(i+1)*parts] += np.linspace(color[0], colorlist[i+1][0], parts)
            greens[i*parts:(i+1)*parts] += np.linspace(color[1], colorlist[i+1][1], parts)
            blues[i*parts:(i+1)*parts] += np.linspace(color[2], colorlist[i+1][2], parts)
        colors[:,0] = reds
        colors[:,1] = greens
        colors[:,2] = blues
        colors[:,3] = np.ones(num)
        return ListedColormap(colors)