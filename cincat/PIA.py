# ============================================================================
#  ________  ___  ________
# |\   __  \|\  \|\   __  \
# \ \  \|\  \ \  \ \  \|\  \
#  \ \   ____\ \  \ \   __  \
#   \ \  \___|\ \  \ \  \ \  \
#    \ \__\    \ \__\ \__\ \__\
#     \|__|     \|__|\|__|\|__|
#
#  ________  ___  ________   ________  ________  _________
# |\   ____\|\  \|\   ___  \|\   ____\|\   __  \|\___   ___\
# \ \  \___|\ \  \ \  \\ \  \ \  \___|\ \  \|\  \|___ \  \_|
#  \ \  \    \ \  \ \  \\ \  \ \  \    \ \   __  \   \ \  \
#   \ \  \____\ \  \ \  \\ \  \ \  \____\ \  \ \  \   \ \  \
#    \ \_______\ \__\ \__\\ \__\ \_______\ \__\ \__\   \ \__\
#     \|_______|\|__|\|__| \|__|\|_______|\|__|\|__|    \|__|
#
# ============================================================================
# Imported packages
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import xlwt
# ============================================================================

# ============================================================================
# Globals
# ============================================================================
t_exp = np.array([0, 5, 10, 20, 30, 45, 65, 90, 125, 160, 213])
c_eto_exp = np.array([7.557, 7.045, 6.614, 5.889, 5.267,
                      4.401, 3.582, 2.742, 2.022, 1.549, 0.803])

r_eto_exp = -np.gradient(c_eto_exp, t_exp, edge_order=2)

k = [0.000347, 0.000111, 0.000152]
c0 = [7.557, 34.0065, 0, 0, 0]


C_ETO_INDEX = 0
C_W_INDEX = 1
C_MEG_INDEX = 2
C_DEG_INDEX = 3
C_TEG_INDEX = 4
C_COUNT = 5

wb = xlwt.Workbook()
sheet = wb.add_sheet("Resultados")
# ============================================================================

def r_c_eto(k, c):
  k_1, k_2, k_3 = k[0], k[1], k[2]

  c_eto = c[C_ETO_INDEX]
  c_w = c[C_W_INDEX]
  c_meg = c[C_MEG_INDEX]
  c_deg = c[C_DEG_INDEX]
  c_teg = c[C_TEG_INDEX]

  return -(k_1 * c_w * c_eto + k_2 * c_meg * c_eto + k_3 * c_deg * c_eto)


def r_c_w(k, c):

  k_1, _, _ = k[0], k[1], k[2]

  c_eto = c[C_ETO_INDEX]
  c_w = c[C_W_INDEX]
  _ = c[C_MEG_INDEX]
  _ = c[C_DEG_INDEX]
  _ = c[C_TEG_INDEX]

  return -(k_1 * c_w * c_eto)


def r_c_meg(k, c):

  k_1, k_2, _ = k[0], k[1], k[2]

  c_eto = c[C_ETO_INDEX]
  c_w = c[C_W_INDEX]
  c_meg = c[C_MEG_INDEX]
  _ = c[C_DEG_INDEX]
  _ = c[C_TEG_INDEX]

  return (k_1 * c_w * c_eto - k_2 * c_meg * c_eto)


def r_c_deg(k, c):

  _, k_2, k_3 = k[0], k[1], k[2]

  c_eto = c[C_ETO_INDEX]
  _ = c[C_W_INDEX]
  c_meg = c[C_MEG_INDEX]
  c_deg = c[C_DEG_INDEX]
  _ = c[C_TEG_INDEX]

  return k_2 * c_meg * c_eto - k_3 * c_deg * c_eto


def r_c_teg(k, c):

  _, _, k_3 = k[0], k[1], k[2]

  c_eto = c[C_ETO_INDEX]
  _ = c[C_W_INDEX]
  _ = c[C_MEG_INDEX]
  c_deg = c[C_DEG_INDEX]
  _ = c[C_TEG_INDEX]

  return k_3 * c_deg * c_eto


def generate_eq_system(k):

  def __f(t, c):
    out = np.empty(C_COUNT)
    out[C_ETO_INDEX] = r_c_eto(k, c)
    out[C_W_INDEX] = r_c_w(k, c)
    out[C_MEG_INDEX] = r_c_meg(k, c)
    out[C_DEG_INDEX] = r_c_deg(k, c)
    out[C_TEG_INDEX] = r_c_teg(k, c)
    return out

  return __f

# NOTE(samuel): avoid calling too often, ofc it's and extremely low call


def solve_at_t_for_k(k, c0, t):
  # Generate the ecuation system with the specified k
  f = generate_eq_system(k)

  # Solve with RK45
  s = solve_ivp(f, [0, t], c0, method="RK45", max_step=0.1)

  # Make sure the last T it's the same as the one we are looking for
  assert s.t[-1] == t

  # Return the "last" concentrations as an array
  return np.array(s.y).transpose()[-1]


# NOTE(samuel): trash code, saves last result and just returns it if it uses
#               the same parameters. reduces minimize time by alot
__compare_c_cache = None
__compare_k_cache = None
__compare_c0_cache = None


def calc_compare_model(k, c0):
  global __compare_c_cache
  global __compare_k_cache
  global __compare_c0_cache

  # check if the model has been previously calculated with the same parameters,
  # if it has, just return the stored values
  if __compare_c0_cache is not None and __compare_c0_cache == c0:
    if __compare_k_cache is not None and __compare_k_cache == k:
      return t_exp, __compare_c_cache

  # initialize an empty array
  c = np.empty((len(t_exp), C_COUNT))

  # store the new model value
  for i, t in enumerate(t_exp):
    c[i] = solve_at_t_for_k(k, c0, t)

  # update the stored values
  __compare_c_cache = c
  __compare_k_cache = k
  __compare_c0_cache = c0

  # return the calculated values
  return t_exp, c


def calc_plotting_model(k, c0):
  # generate the ecuation system
  f = generate_eq_system(k)

  # as there is no regard for spacing just solve using solve_ivp
  r = solve_ivp(f, [0, t_exp[-1]], c0, method="RK45", max_step=0.1)

  # return the values
  return r.t, r.y


def calc_r2_model(k, kr, r2):
  # get the model t and c values
  t, c = calc_compare_model(k, c0)

  # calculate the r values with the concentrations
  return np.array([r2(kr, cc) for cc in c])


def r2_0(kr, c):
  k_obs, k_1, k_3 = kr[0], kr[1], kr[2]

  c_eto = c[C_ETO_INDEX]
  c_w = c[C_W_INDEX]
  c_meg = c[C_MEG_INDEX]
  _ = c[C_DEG_INDEX]
  _ = c[C_TEG_INDEX]

  return k_obs * (c_eto * c_w) / (1 + k_1 * c_eto + k_3 * c_meg)


def r2_1(kr, c):

  k_obs, _, _ = kr[0], kr[1], kr[2]

  _ = c[C_ETO_INDEX]
  c_w = c[C_W_INDEX]
  _ = c[C_MEG_INDEX]
  _ = c[C_DEG_INDEX]
  _ = c[C_TEG_INDEX]

  return k_obs * c_w


def r2_2(kr, c):
  
  k_obs, k_1, k_3 = kr[0], kr[1], kr[2]

  c_eto = c[C_ETO_INDEX]
  c_w = c[C_W_INDEX]
  c_meg = c[C_MEG_INDEX]
  _ = c[C_DEG_INDEX]
  _ = c[C_TEG_INDEX]

  return k_obs * c_eto * c_w / (k_3 * c_meg)


def r2_3(kr, c):
  
  k_obs, k_1, k_3 = kr[0], kr[1], kr[2]

  c_eto = c[C_ETO_INDEX]
  c_w = c[C_W_INDEX]
  c_meg = c[C_MEG_INDEX]
  _ = c[C_DEG_INDEX]
  _ = c[C_TEG_INDEX]

  return k_obs * c_eto * c_w / (1 + k_3 * c_meg)


def r2_4(kr, c):

  k_obs, k_1, k_3 = kr[0], kr[1], kr[2]

  c_eto = c[C_ETO_INDEX]
  c_w = c[C_W_INDEX]
  c_meg = c[C_MEG_INDEX]
  _ = c[C_DEG_INDEX]
  _ = c[C_TEG_INDEX]

  return k_obs * c_eto * c_w / (1 + k_1 * c_eto)
  

def msqr(mod, exp):
  return np.sum((mod - exp) * (mod - exp))


def are(mod, exp):
  return np.average(np.abs((mod - exp) / exp) * 100)


def minimize_r2_method(k, kr0, r2, ev=msqr):

  def __f(kr):
    return ev(r_eto_exp, calc_r2_model(k, kr, r2))

  def __not_negative(v):
    return np.sum([0 > vi for vi in v])
  
  constraints = ({"type": "eq", "fun": __not_negative})

  return minimize(__f, kr0, constraints=constraints)


def calc_model_errors(k, c0):
  _, c = calc_compare_model(k, c0)

  err = np.empty(len(c_eto_exp))

  model = c.transpose()[C_ETO_INDEX]

  for i, (c_mod, c_exp) in enumerate(zip(model, c_eto_exp)):
    err[i] = np.abs((c_exp - c_mod) / c_exp) * 100

  return err


def calc_r2_error(kr, r2):
  model = calc_r2_model(k, kr, r2)

  err = np.empty(len(c_eto_exp))

  for i, (r_mod, r_exp) in enumerate(zip(model, r_eto_exp)):
    err[i] = np.abs((r_exp - r_mod) / r_exp) * 100

  return err


__write_to_excel_offset = 0
def write_row_excel(values):
  global __write_to_excel_offset

  for i, v in enumerate(values):
    sheet.write(__write_to_excel_offset, i, v)

  __write_to_excel_offset += 1
  
def save_excel():
  wb.save("PIA.xls")


def display_integrated_model_result(k, c0):
  t, c = calc_compare_model(k, c0)
  t_plot, c_plot = calc_plotting_model(k, c0) 
  err = calc_model_errors(k, c0)

  pt = PrettyTable()
  pt.field_names = ["t (min)", "C_EtO experimental", "C_EtO modelo", "% error"]

  for ti, c_exp, c_mod, e in zip(t, c_eto_exp, c, err):
    pt.add_row([ti, c_exp, c_mod[C_ETO_INDEX], e])

  print(pt)

  write_row_excel(["t (min)", "C_EtO experimental", "C_EtO modelo", "% error"])
  for ti, c_exp, c_mod, e in zip(t, c_eto_exp, c, err):
    write_row_excel([float(ti), float(c_exp), float(c_mod[C_ETO_INDEX]), float(e)])

  write_row_excel([])
  save_excel()


  plt.plot(t_exp, c_eto_exp, "o", label="$C_{EtO}$ experimental")
  plt.plot(t_plot, c_plot[C_ETO_INDEX], label="$C_{EtO}$ modelado")
  plt.legend()
  plt.xlabel("$t (min)$")
  plt.ylabel("$C_{EtO} (mol / L)$")
  plt.grid()
  plt.show()

  plt.plot(t_plot, c_plot[C_ETO_INDEX], label="$C_{EtO}$")
  plt.plot(t_plot, c_plot[C_MEG_INDEX], label="$C_{MEG}$")
  plt.plot(t_plot, c_plot[C_W_INDEX], label="$C_{W}$")
  plt.xlabel("$t (min)$")
  plt.ylabel("$C (mol / L)$")
  plt.grid()
  plt.legend()
  plt.show()

def display_minimize_r2_result(r2, title=""):
  r = minimize_r2_method(k, [0.001149, 0.299284, 0.31144], r2, msqr)
  model = calc_r2_model(k, r.x, r2)
  err = calc_r2_error(r.x, r2)

  pt = PrettyTable()
  pt.field_names = ["k", "valor"]
  pt.add_row(["k_obs", r.x[0]])
  pt.add_row(["k_1", r.x[1]])
  pt.add_row(["k_2", r.x[2]])
  print(pt)

  pt = PrettyTable()
  pt.field_names = ["t (min)", "r_EtO experimental", "r_EtO modelo", "% error"]

  for ti, r_exp, r_mod, e in zip(t_exp, r_eto_exp, model, err):
    pt.add_row([ti, r_exp, r_mod, e])

  print(pt)

  write_row_excel(["t (min)", "r_EtO experimental", "r_EtO modelo", "% error"])
  for ti, r_exp, r_mod, e in zip(t_exp, r_eto_exp, model, err):
    write_row_excel([float(ti), float(r_exp), float(r_mod), float(e)])

  write_row_excel([])
  save_excel()

  plt.plot(t_exp, r_eto_exp, "o", label="$-r_{EtO}$ experimental")
  plt.plot(t_exp, model, label="$-r_{EtO}$ modelo")
  plt.xlabel("$t (min)$")
  plt.ylabel("$-r_{EtO}$ (L / min)")
  plt.legend()
  plt.grid()
  plt.show()


if __name__ == "__main__":
  display_minimize_r2_result(r2_0)
  display_minimize_r2_result(r2_1)
  display_minimize_r2_result(r2_2)
  display_minimize_r2_result(r2_3)
  display_minimize_r2_result(r2_4)

  display_integrated_model_result(k, c0)

