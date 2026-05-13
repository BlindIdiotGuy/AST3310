import numpy as np

Nx = np.linspace(0, 19, 20)
Ny = np.linspace(0, 9, 10)
x, y = np.meshgrid(Nx, Ny)
print(y)