import numpy as np
from scipy.optimize import minimize

n = 0.7

v0 = 100  # L / min
yA0 = 0.3
yI0 = 0.7
T = 200 + 273.15  # ˚K
P = 1.5  # atm

cA0 = yA0 * P / (T * 0.08206)

# V1 = Fa0 * XA / -rA
epsilon = 0.3

xA2 = 0.8
k = 0.3
FA0 = v0 * cA0


def cA(xA):
  return cA0 * (1 - xA) / (1 + epsilon * xA)


def v1(xA1):
  return FA0 * (xA1 - 0) / (k * np.power(cA(xA1), n))


def v2(xA1):
  return FA0 * (xA2 - xA1) / (k * np.power(cA(xA2), n))


def v(xA1):
  return v1(xA1) + v2(xA1)


r = minimize(v, 0.5, method="Nelder-Mead")
print(f"P1: {r}")

n = 1
v0 = 120  # L / min
k = 3.5
xAf = 0.6375

FA0 = v0


def v_pfr(xA1):
  return -v0 / k * np.log(1 - xA1)


def v_cstr(xA1):
  return v0 / k * (xAf - xA1) / (1 - xAf)


def v(xA1):
  return v_cstr(xA1) + v_pfr(xA1)


print(v_pfr(0.75))


def fobj(xA1):
  return np.power(v_pfr(0.75) - v(xA1[0]), 2)


bounds = ((0.0, 0), (1.0, 0))
r = minimize(v, (0.229, 0), method="TNC", bounds=bounds)
print(r)
