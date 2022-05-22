import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import xlwt


x_elv = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1]
y_elv = [0, 0.21, 0.37, 0.51, 0.65, 0.72, 0.79, 0.86, 0.91, 0.96, 0.98, 1]
x_D = 0.96
z_F = 0.5

# simple hack to analyze function calls
def wrap_result_print(f):
  def _f(*args):
    print("args: ", *args)
    v = f(*args)
    print(f"v: {v}")
    return v

  return _f


def find_closest_indices(l, v):
  for i, lv in enumerate(l[:-1]):
    if lv < v and l[i + 1] > v:
      return i, i + 1

    if lv == v:
      return i, i

    if l[i + 1] > v:
      return i + 1, i + 1

  raise Exception(f"index not found l: {lv}, v: {v}")


def create_eq_from_points(p1, p2):
  m = (p2[1] - p1[1]) / (p2[0] - p1[0])
  b = p1[1] - m * p1[0]

  return Polynomial([float(b), float(m)])


# because we are dealing with a saturated liquid, we just need to find were
# in the vle curve and the straight line in x intersect
def find_vle_inter_at_x(lx, ly, x):
  i, j = find_closest_indices(lx, x)

  # the exact point is in the list
  if i == j:
      return x, ly[i]

  return x, create_eq_from_points((lx[i], ly[i]), (lx[j], ly[j]))(x)


def find_vle_inter_at_y(lx, ly, y):
  i, j = find_closest_indices(ly, y)

  if i == j:
    return lx[i], ly[i]

  m = (ly[j] - ly[i]) / (lx[j] - lx[i])
  b = ly[i] - m * lx[i]

  return (y - b) / m, y


def _gen_mccabe_points(x, y, eq, start_x, start_y, end_x):
  assert 1 == eq.degree()

  p = []

  cur_x = start_x
  cur_y = start_y

  while cur_x > end_x:
    cur_x, _ = find_vle_inter_at_y(x, y, cur_y)
    p.append((cur_x, cur_y))

    # cur_y = wrap_result_print(eq)(cur_x)
    cur_y = eq(cur_x)
    p.append((cur_x, cur_y))

  return p


def gen_mccabe_points(x, y, eq1, eq2, start_x, mid_x, end_x):
  p = [(start_x, start_x)]
  p.extend(_gen_mccabe_points(x, y, eq1, start_x, start_x, mid_x))
  p.pop()
  p.extend(_gen_mccabe_points(x, y, eq2, p[-1][0], p[-1][1], end_x))
  return p


def gen_destil_eq(x_D, R):
  a0 = x_D / (R + 1)
  a1 = R / (R + 1)
  return Polynomial([a0, a1])


def gen_feed_eq(z_F, q):
  a0 = z_F / (1 - q)
  a1 = q / (q - 1)
  return Polynomial([a0, a1])


def calc_rmin(x, y, z_F, x_D):
  x_inter, y_inter = find_vle_inter_at_x(x, y, z_F)

  p1 = (x_D, x_D)
  p2 = (x_inter, y_inter)

  # in theory we could just gen the eq from this, but w/e
  eq = create_eq_from_points(p1, p2)

  # y intercept at x = 0
  b = eq.coef[0]

  # x_D / (R + 1) = b
  # R + 1 = x_D / b
  # R = b / x_D - 1

  return x_D / b - 1


def calc_rop(x, y, z_F, x_D):
  return 2 * calc_rmin(x, y, z_F, x_D)


def gen_eq_from_data(x, y, z_F, x_D, x_B):
  R = calc_rop(x, y, z_F, x_D)

  eq_destil = gen_destil_eq(x_D, R)
  eq_reheat = create_eq_from_points((x_B, x_B), (z_F, eq_destil(z_F)))

  return eq_destil, eq_reheat


def calc_intercept(eq_1, eq_2):
  assert 1 == eq_1.degree() and 1 == eq_2.degree()

  a = np.array([[-eq_1.coef[1], 1], [-eq_2.coef[1], 1]])
  b = np.array([eq_1.coef[0], eq_2.coef[0]])

  return np.linalg.solve(a, b)


def create_steps_generator(x, y, z_F, x_D):

  def _f(x_B):
    eq_destil, eq_reheat = gen_eq_from_data(x, y, z_F, x_D, x_B)
    x_inter, y_inter = calc_intercept(eq_destil, eq_reheat)
    return gen_mccabe_points(x, y, eq_destil, eq_reheat, x_D, x_inter, x_B)

  return _f


def get_mccabe_steps(gen, x):
  return len(gen(x)) / 2 - 1


def find_close_x_B(gen, req_steps, x_B):
  max_iter = 10000
  # not use a while, as looping to infinity sound like a bad idea
  for i in range(max_iter):
    steps = gen(x_B)
    cur_steps = get_mccabe_steps(gen, x_B)

    if cur_steps > req_steps:
      x_B = steps[req_steps][0]

    if cur_steps < req_steps:
      x_B = 0.999 * steps[-1][0]

    if req_steps == cur_steps:
      break

  # bad, super slow and has a huge error
  dx = 1e-3

  while req_steps == get_mccabe_steps(gen, x_B):
    x_B = x_B - dx

  # undo the last change to undo the extra step
  return x_B + dx


gen = create_steps_generator(x_elv, y_elv, z_F, x_D)

x_B = find_close_x_B(gen, 7, 0.1)

steps = gen(x_B)
stepst = np.array(steps).transpose()

eq_destil, eq_reheat = gen_eq_from_data(x_elv, y_elv, z_F, x_D, x_B)
x_inter, y_inter = calc_intercept(eq_destil, eq_reheat)

x_reheat = np.array([x_B, x_inter])
x_destil = np.array([x_D, x_inter])
x_feed = np.array([z_F, x_inter])

plt.grid()
plt.plot([0, 1], [0, 1])
plt.plot(stepst[0], stepst[1])
plt.plot(x_elv, y_elv)
plt.plot(x_feed, [z_F, y_inter])
plt.plot(x_reheat, eq_reheat(x_reheat))
plt.plot(x_destil, eq_destil(x_destil))
plt.show()

wb = xlwt.Workbook()

sheet1 = wb.add_sheet("Sheet 1")

for i, step in enumerate(steps):
  sheet1.write(i, 0, step[0])
  sheet1.write(i, 1, step[1])

wb.save("data.xls")
