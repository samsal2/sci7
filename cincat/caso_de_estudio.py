import numpy as np
from scipy.integrate import RK45, solve_ivp
import matplotlib.pyplot as plt

def r_pt(k1, c_pt, c_eh):
  return -k1 * c_pt * c_eh


def r_eh(k1, k2, c_pt, c_eh, c_me):
  return -k1 * c_pt * c_eh - k2 * c_me * c_eh


def r_me(k1, k2, c_pt, c_eh, c_me):
  return k1 * c_pt * c_eh - k2 * c_me * c_eh


def r_dehtl(k2, c_eh, c_me):
  return k2 * c_me * c_eh


def gen_r(k1, k2):

  def __f(t, c):
    c_pt = c[0]
    c_eh = c[1]
    c_me = c[2]
    r0 = r_pt(k1, c_pt, c_eh)
    r1 = r_eh(k1, k2, c_pt, c_eh, c_me)
    r2 = r_me(k1, k2, c_pt, c_eh, c_me)
    r3 = r_dehtl(k2, c_eh, c_me)
    return r0, r1, r2, r3

  return __f

def solve_for_k1_and_k2_rk45(k1, k2, c0, start, end):
  solution = RK45(gen_r(k1, k2), start, c0, end)

  t = []
  c = []

  while 'finished' != solution.status:
    solution.step();
    t.append(solution.t)
    c.append(solution.y)

  return np.array(t), np.array(c).transpose()


def solve_for_k1_and_k2(k1, k2, c0, start, end):
  solution = solve_ivp(gen_r(k1, k2), (start, end), c0, method="RK45")
  return np.array(solution.t), solution.y


c_pt_0 = 0.8
c_eh_0 = 2.1
  
def k1(T, c_pt_0=c_pt_0):
  return np.exp(-10620 / T + 17.305) / c_pt_0


def k2(T, c_pt_0=c_pt_0):
  return np.exp(-4188.3 / T + 6.230) / c_pt_0


def solve_for_T(T, c0, start, end):
  return solve_for_k1_and_k2(k1(T), k2(T), c0, start, end)

class SolutionRange:
  def __init__(self, T, t):
    self.T = T
    self.t = t

  def __str__(self):
    return f"T: {self.T} t: {self.t}"

  def __repr__(self):
    return str(self)


def solve_for_n_solution_ranges(ranges, c):
  tl = np.array([])
  cl = np.array([[], [], [], []])
  for r in ranges:
    t, cn = solve_for_T(r.T, c, r.t[0], r.t[-1])
  
    tl = np.concatenate((tl, t), axis=0)
    cl = np.concatenate((cl, cn), axis=1)
    c = [cn[0][-1], cn[1][-1], cn[2][-1], cn[3][-1]]
  return tl, cl


t_exp_min = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 140, 180, 210, 230, 250])
T_exp_K = np.array([453.2, 455.2, 457.2, 457.2, 458.2, 459.2, 461.2, 462.2, 463.2, 464.2, 466.2, 475.2, 483.2, 491.2, 503.2, 519.2])
c_pt_exp_M = np.array([1, 0.934, 0.849, 0.791, 0.751, 0.703, 0.661, 0.621, 0.586, 0.553, 0.518, 0.377, 0.259, 0.177, 0.121, 0.068]) * c_pt_0
c_eh_exp_M = np.array([2.7, 2.617, 2.450, 2.325, 2.242, 2.144, 2.061, 1.978, 1.908, 1.839, 1.763, 1.478, 1.242, 1.075, 0.964, 0.860]) * c_pt_0
c_me_exp_M = np.array([0, 0.049, 0.052, 0.043, 0.039, 0.039, 0.039, 0.037, 0.037, 0.032, 0.027, 0.024, 0.023, 0.022, 0.022, 0.024]) * c_pt_0
c_dehtl_exp_M = np.array([0, 0.017, 0.099, 0.166, 0.210, 0.258, 0.300, 0.343, 0.378, 0.414, 0.455, 0.599, 0.718, 0.802, 0.857, 0.908]) * c_pt_0

def model(t, T_off=0, print_k=False):
  tin = [T_exp_K[0] + T_off,
         T_exp_K[3] + T_off,
         T_exp_K[6] + T_off,
         T_exp_K[9] + T_off,
         T_exp_K[12] + T_off]
  din = [[t_exp_min[0], t_exp_min[3]],
         [t_exp_min[3], t_exp_min[6]],
         [t_exp_min[6], t_exp_min[9]],
         [t_exp_min[9], t_exp_min[12]],
         [t_exp_min[12], t_exp_min[15]]]

  if False:
    print(f"t{din}")

  if print_k:
    for T in tin:
      rk1 = k1(T)
      rk2 = k2(T)
      print(f"T: {T} k1: {rk1} k2: {rk2} k1 / k2: {rk1 / rk2} k2 / k1:  {rk2 / rk1}")

  tout = []
  dout = []

  for ti, d in zip(tin, din):
    o = [None, None]
 
    if t[1] < d[0]:
      break

    if t[1] >= d[0]:
      o[0] = d[0]
  
    if t[1] < d[1]:
      o[1] = t[1]
    else:
      o[1] = d[1]

    dout.append(o)
    tout.append(ti)
   
  ranges = []            
  for ti, d in zip(tout, dout):
    ranges.append(SolutionRange(ti, d))

  return solve_for_n_solution_ranges(ranges, [c_pt_0, c_eh_0, 0, 0]) 
   
  
def residuals(print_k=False):
  c = np.array([[c_pt_0], [c_eh_0], [0] ,[0]])

  for i, t in enumerate(t_exp_min):
    if 0 == t:
      continue

    tm, nc = model([0, t], 0, print_k)

    assert tm[-1] == t

    res0 = nc[0][-1] - c_pt_exp_M[i]
    res1 = nc[1][-1] - c_eh_exp_M[i]
    res2 = nc[2][-1] - c_me_exp_M[i]
    res3 = nc[3][-1] - c_dehtl_exp_M[i]

    c = np.concatenate((c, np.array([[res0], [res1], [res2], [res3]])), axis=1)

  return t_exp_min, c

# t, c = solve_for_T(T_exp_K[0], [c_pt_0, c_eh_0, 0, 0], 0, t_exp_min[-1])
t, c = model([t_exp_min[0], t_exp_min[-1]])

plt.plot(t_exp_min, c_pt_exp_M, ".r", label="$C_{PT}$ experimental")
plt.plot(t_exp_min, c_eh_exp_M, ".g", label="$C_{EH}$ experimental")
plt.plot(t_exp_min, c_me_exp_M, ".b", label="$C_{ME}$ experimental")
plt.plot(t_exp_min, c_dehtl_exp_M, ".m", label="$C_{DEHTL}$ experimental")

plt.title("Modelo vs Experimental")
plt.plot(t, c[0], "-r", label="$C_{PT}$ calculada")
plt.plot(t, c[1], "-g", label="$C_{EH}$ calculada")
plt.plot(t, c[2], "-b", label="$C_{ME}$ calculada")
plt.plot(t, c[3], "-m", label="$C_{DEHTL}$ calculada")
plt.legend()
plt.xlabel("t (min)")
plt.ylabel("C (M)")
plt.grid()
plt.show()

# sens -30 ˚K
tm30, cm30 = model([t_exp_min[0], t_exp_min[-1]], -30, True)
print("------------------------------------")
tp30, cp30 = model([t_exp_min[0], t_exp_min[-1]], 30, True)

plt.title("Sensibilidad $C_{DEHTL}$")
plt.plot(tm30, cm30[3], label="$C_{DEHTL}$ a $+30 ˚K$")
plt.plot(tp30, cp30[3], label="$C_{DEHTL}$ a $-30 ˚K$")
plt.plot(t, c[3], label="$C_{DEHTL}$")
plt.xlabel("t (min)")
plt.ylabel("$C_{DEHTL}$ (M)")
plt.legend()
plt.grid()
plt.show()

plt.title("Sensibilidad $C_{ME}$")
plt.plot(tm30, cm30[2], label="$C_{ME}$ a $+30 ˚K$")
plt.plot(tp30, cp30[2], label="$C_{ME}$ a $-30 ˚K$")
plt.plot(t, c[2], label="$C_{ME}$")
plt.xlabel("t (min)")
plt.ylabel("$C_{ME}$ (M)")
plt.legend()
plt.grid()
plt.show()

_, res = residuals()

rmsep1 = np.sqrt(np.average(res * res))
print(rmsep1)

plt.title("Residuales $C_{ME}$")
plt.plot(t_exp_min, res[2], ".b")
plt.xlabel("t (min)")
plt.ylabel("$C_{ME}$ (M)")
plt.grid()
plt.show()

plt.title("Residuales $C_{DEHTL}$")
plt.plot(t_exp_min, res[3], ".b")
plt.xlabel("t (min)")
plt.ylabel("$C_{DEHTL}$ (M)")
plt.grid()
plt.show()



