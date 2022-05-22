import numpy as np
from scipy.optimize import minimize


v = 3.05
Y_p = 0.01
Y_s = 0.031
T_G = 65.6
V_h = (2.83e-3 + 4.56e-3 * Y_p) * (T_G + 273.15)
rho = (1 + Y_p) / V_h
G = v * rho * 3600

h_c = 0.0204 * np.power(G, 0.8)
C_s = (1.005 + 1.88 * Y_p) * 1e3
k_y = h_c / C_s
epsilon = 0.95


T_r = 93.3
T_s_0 = 32.2

water_heat_table = \
  [[25, 30, 35],
   [2442, 2431, 2419]] 

def h_r(T_r, T_s):
  T_r_K = T_r + 273.15
  T_s_K = T_s + 273.15
  return 5.279e-8 * epsilon * (np.power(T_r_K, 4) - np.power(T_s_K, 4)) / (T_r - T_s)

print(h_r(T_r, 32.2))


def water_heat_find_interval(T):
  data = np.array(water_heat_table).transpose()

  for i, d in enumerate(data[:-1]):
    l = d[0]
    h = data[i + 1][0]

    if T > l and T < h:
      return d, data[i + 1]

  raise RuntimeError(f"No T interval found for {T}")
    

def water_heat(T):
  l, h = water_heat_find_interval(T)
  return l[1] + (h[1] - l[1]) / (h[0] - l[0]) * (T - l[0])


def fobj(T_s):
  hr = h_r(T_r, T_s)
  heat = water_heat(T_s)
  lhs = (Y_s - Y_p) * heat * 1e3 / C_s
  rhs = (T_G - T_s) + hr / h_c * (T_r - T_s)
  return (lhs - rhs) * (lhs - rhs)

print(minimize(fobj, T_s_0))


