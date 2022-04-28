from scipy.optimize import curve_fit
import numpy as np
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def strip(v):
  return v.magnitude


def gen_model(CA0, deg, k):
  assert 1 != deg
  n = deg - 1

  def __f(t):
    return np.power(-1 / (-k * t - 1 / np.power(CA0, n)), 1 / n)

  return __f

# Rxn: A -> 2R


vA = 1
vR = 2

T = Q_(200, ureg.celsius)
PT0 = Q_(1, ureg.atm)

yA0 = 0.5
CA0 = (1 * yA0) / (0.08205 * 473)

t = Q_([5, 15, 30, 40, 50, 80], ureg.min)
PT = Q_([15.5, 16.7, 17.9, 18.4, 18.8, 19.6], ureg.psi)

epsilon = (vR / vA - vA / vA) * yA0

CA = CA0 * (1 - (PT - PT0) / (epsilon * PT0))

# Probando primer orden
# ln(CA / CA) = -kt

X = strip(-t)
Y = strip(np.log(CA / CA0))

params = curve_fit(lambda t, k: k * t, X, Y)
k = params[0]

print(f"k primer orden {k}")

CA_mod = CA0 * np.exp(-k * strip(t))
err = sum(np.abs((CA - CA_mod) / CA))
print(f"err primer orden {strip(err)}\n")

# Probando segundo orden
# -1 / CA + 1 / CA0 = -kt

X = strip(-t)
Y = strip(-1 / CA + 1 / CA0)

params = curve_fit(lambda t, k: k * t, X, Y)
k = params[0]

print(f"k segundo orden {k}")

k_segundo_orden = k[0]

CA_mod = -1 / (-k * strip(t) - 1 / CA0)

err = sum(np.abs((CA - CA_mod) / CA))
print(f"err segundo orden {strip(err)}\n")


# Probando tercer orden
# -1 / CA + 1 / CA0^2 = -kt

X = strip(-t)
Y = strip(-1 / np.power(CA, 2) + 1 / np.power(CA0, 2))

params = curve_fit(lambda t, k: k * t, X, Y)
k = params[0]

print(f"k tercer orden {k}")

CA_mod = np.sqrt(-1 / (-k * strip(t) - 1 / np.power(CA0, 2)))
err = sum(np.abs((CA - CA_mod) / CA))
print(f"err tercer orden {strip(err)}")

# b)
model = gen_model(CA0, 2, k_segundo_orden)

# f)

# el segundo orden fue el mejor
print(f"\na) -rA = {k_segundo_orden} * Ca^2")
print(f"b) CA a 1hr = {model(60)}")
