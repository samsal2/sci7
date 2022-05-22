import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import minimize

rA = np.array([0.00001, 0.000005, 0.000002, 0.000001])
XA = np.array([0, 0.2, 0.4, 0.6])
inv = 1 / rA

XA1 = 0.3

T = 500.15
P = 10
ya0 = 0.333

CA0 = ya0 * P / (T * 0.08206)
f = Polynomial.fit(XA, inv, 2).convert()
finteg = f.integ()

v = 33.3333
FA0 = CA0 * v
V = FA0 * (finteg(XA1) - finteg(0))

print(f"P1: {V}")

XA2 = 0.5

R = FA0 * (XA2 - XA1) * f(XA2) 
print(f"P2: {R}")

R = FA0 * (0.8 - 0) * f(0.8) 
print(f"P3: {R}")

R = FA0 * (0.6 - 0.5) * f(0.6)
print(f"P4: {R}")

CA0 = 1
FA0 = 1
XA1 = 0.9
  
def fobj(XA2):
  lhs = (XA1 - 0) / ((1 - XA1) * (1 - XA1))
  rhs = (XA2 - XA1) / ((1 - XA2) * (1 - XA2))
  return (lhs - rhs) * (lhs - rhs)

r = minimize(fobj, 0.9)
print(f"P5: {r}")

