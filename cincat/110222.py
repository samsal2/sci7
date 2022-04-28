import numpy as np
from scipy.integrate import RK45

def gen_solver(k_1, k_3, t_0=0, t_boundary=100, y_0=np.array([1, 0, 0])):

  def __f(t, y):
    r0 = -k_1 * y[0] - k_3 * y[1]
    r1 = k_1 * y[0] - k_3 * y[1]
    r2 = k_1 * y[0] + 2 * k_3 * y[1]
    return np.array([r0, r1, r2])

  return RK45(__f, t_0, y_0, t_boundary)
 

def gen_data(solver, n):

  def __d():
    solver.step()
    return solver.t, solver.y

  return np.array([__d() for i in range(n)])

   
sol = gen_solver(1000, 1000)

  
print(gen_data(sol, 100))



