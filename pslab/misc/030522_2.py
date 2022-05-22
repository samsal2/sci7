import numpy as np
from scipy.optimize import minimize

C_s = 1035.08
h_r = 9
h_c = 30
U_k = 10
T_r = 110
T_s_0 = 30
Y_s = 0.016
T_G = 100

water_heat_table = \
  [[10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
   [2478, 2466, 2454, 2442, 2431, 2419, 2407, 2395, 2383, 2371]] 

def water_heat_find_interval(T):
  data = np.array(water_heat_table).transpose()

  for i, d in enumerate(data[:-1]):
    l = d[0]
    h = data[i + 1][0]

    if T >= l and T <= h:
      return d, data[i + 1]

  raise RuntimeError(f"No T interval found for {T}")
    

def water_heat(T):
  l, h = water_heat_find_interval(T)
  return l[1] + (h[1] - l[1]) / (h[0] - l[0]) * (T - l[0])


def next_T_s(Y, T_s):
  heat = water_heat(T_s)

  a = (Y_s - Y) * heat * 1e3 / C_s
  b = (1 + U_k / h_c)
  c =  h_r / h_c

  print(f"lhs: {a}, rhs {b * (T_G - T_s) + c * (T_r - T_s)}")
  

  return - (a - b - c) / (b + c)

T = [T_s_0]

while True:
  Y = float(input(f"Next Y (T = {T[-1]})?: "))

  T.append(next_T_s(Y, T[-1]))

  if abs((T[-1] - T[-2]) / T[-1]) * 100 < 5e-2:
    break;


print(T)


