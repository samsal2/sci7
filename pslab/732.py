import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import xlwt


def calc_intercept(eq_1, eq_2):
  assert 1 == eq_1.degree() and 1 == eq_2.degree()

  # y = m_1 * x + b_1
  # y = m_2 * x + b_2
  # y - m_1 * x = b_1
  # y - m_2 * x = b_2
  # |-m_1, 1| |x|   |b_1|
  # |_m_2, 1| |y| = |b_2|

  a = np.array([[-eq_1.coef[1], 1], [-eq_2.coef[1], 1]])
  b = np.array([eq_1.coef[0], eq_2.coef[0]])

  return np.linalg.solve(a, b)


def find_closest_indices(l, v):
  for i, lv in enumerate(l[:-1]):
    if lv < v and l[i + 1] > v:
      return i, i + 1

    if lv == v:
      return i, i

    if l[i + 1] == v:
      return i + 1, i + 1

  raise Exception(f"index not found l: {lv}, v: {v}")


def find_intercept_in_list_at_y(lx, ly, y):
  i, j = find_closest_indices(ly, y)

  # not handling this case atm
  assert i != j

  m = (ly[j] - ly[i]) / (lx[j] - lx[i])
  b = ly[i] - m * lx[i]

  return (y - b) / m, y


def _gen_mccabe_points(x, y, eq, start_x, start_y, end_x):
  assert 1 == eq.degree()

  p = []

  cur_x = start_x
  cur_y = start_y

  while cur_x > end_x:
    cur_x, _ = find_intercept_in_list_at_y(x, y, cur_y)
    p.append((cur_x, cur_y))

    cur_y = eq(cur_x)
    p.append((cur_x, cur_y))

  return p


def gen_mccabe_points(x, y, eq1, eq2, start_x, mid_x, end_x):
  p = [(start_x, start_x)]
  p.extend(_gen_mccabe_points(x, y, eq1, start_x, start_x, mid_x))
  p.pop()
  p.extend(_gen_mccabe_points(x, y, eq2, p[-1][0], p[-1][1], end_x))
  return p


def get_mccabe_min_steps(x, y, eq1, start_x, mid_x):
  return (len(_gen_mccabe_points(x, y, eq1, start_x, start_x, mid_x)) + 1) // 2


F = 100  # mol / h
z_F = 0.12  # -
R = 3
x_D = 0.85
D = 0.9 * z_F * F / x_D
L = R * D
B = L + F
x_B = (z_F * F - x_D * D) / B

eq_enriquecimiento = Polynomial([1 / (R + 1) * x_D, R / (R + 1)])

# Sacado a ojo
x_agotamiento_2, y_agotamiento_2 = z_F, 0.304

m_agotamiento = (y_agotamiento_2 - 0) / (x_agotamiento_2 - x_B)
b_agotamiento = y_agotamiento_2 - m_agotamiento * x_agotamiento_2
eq_agotamiento = Polynomial([b_agotamiento, m_agotamiento])

x_enriquecimiento = np.array([x_D, x_agotamiento_2])
y_enriquecimiento = eq_enriquecimiento(x_enriquecimiento)

x_agotamiento = np.array([x_B, x_agotamiento_2])
y_agotamiento = eq_agotamiento(x_agotamiento)

x_alimentacion = np.array([z_F, x_agotamiento_2])
y_alimentacion = np.array([z_F, y_agotamiento_2])

x_elv = [0, 0.019, 0.0721, 0.0966, 0.1238, 0.1661, 0.2337, 0.2608, 0.3273,
         0.3965, 0.5198, 0.5732, 0.6763, 0.7472, 0.8943, 1]

y_elv = [0, 0.17, 0.3891, 0.4375, 0.4704, 0.5089, 0.5445, 0.558, 0.5826,
         0.6122, 0.6599, 0.6841, 0.7385, 0.7815, 0.8943, 1]

steps = gen_mccabe_points(
    x_elv, y_elv, eq_enriquecimiento, eq_agotamiento, x_D, x_agotamiento_2, x_B)

etapas = get_mccabe_min_steps(
    x_elv, y_elv, eq_enriquecimiento, x_D, x_agotamiento_2)

print(etapas)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

major_ticks = np.arange(0, 1, 0.1)
minor_ticks = np.arange(0, 1, 0.1)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which="both")

stepst = np.array(steps).transpose()

plt.plot(stepst[0], stepst[1], "b", label="Etapas")
plt.plot([0, 1], [0, 1], label="45˚")
plt.plot(x_elv, y_elv, label="ELV")
plt.plot(x_enriquecimiento, y_enriquecimiento, label="Enriquecimiento")
plt.plot(x_alimentacion, y_alimentacion, label="Alimentación")
plt.plot(x_agotamiento, y_agotamiento, label="Agotamiento")
plt.xlim(0, 1)
plt.ylim(0, 1)


plt.xlabel("x")
plt.ylabel("y")

plt.legend()
plt.show()

# write the values on an excel sheet

wb = xlwt.Workbook()

sheet1 = wb.add_sheet("Sheet 1")

for i, step in enumerate(steps[:-1]):
  sheet1.write(i, 0, step[0])
  sheet1.write(i, 1, step[1])

wb.save("data.xls")
