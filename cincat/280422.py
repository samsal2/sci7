import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

def foo(a):
  return a


r = np.array([0.0127, 0.012, 0.0114, 0.0244, 0.0232, 0.022, 0.0352, 0.0334, 0.0317])
p_C_atm = np.array([0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6])
p_B_atm = np.array([0, 0.1, 0.2, 0, 0.1, 0.2, 0, 0.1, 0.2])

X1 = p_B_atm
X2 = p_C_atm

y = np.sqrt(p_C_atm / r)

x = np.array([X1, X2]).transpose()
x = np.c_[x, np.ones(x.shape[0])]
result = np.linalg.lstsq(x, y, rcond=None)[0]

@np.vectorize
def model(p_B, p_C):
  v = (result[2] + result[1] * p_C + + result[0] * p_B)
  return 1 / (v * v) * p_C

Y = np.linspace(0.2, 0.6)
X = np.linspace(0, 0.2)

X, Y = np.meshgrid(X, Y)
Z = model(X, Y)

fig = plt.figure()
ax = plt.axes(projection="3d")
msurf = ax.plot_surface(X, Y, Z, cmap="viridis", label="modelo")
esurf = ax.scatter3D(p_B_atm, p_C_atm, r, cmap="Green", label="experimental")
msurf._edgecolors2d = msurf._edgecolor3d
msurf._facecolors2d = msurf._facecolor3d

ax.set_xlabel('$P_B$')
ax.set_ylabel('$P_C$"')
ax.set_zlabel('$r$');
ax.legend()
plt.show()
