import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad

CA0 = 2.2
XAF = 0.7
k = 0.272


def fobj(XA):
  lhs = 1 / (k * CA0 * CA0 * (1 - XA) * XA)
  a = 1 / (k * CA0 * CA0)
  b = np.log((XAF * (1 - XA)) / (XA * (1 - XAF)))
  c = XAF - XA
  rhs = a * b / c
  return lhs - rhs


print(XA := fsolve(fobj, 0.1))


def fobj2(R):
  return (R / (R + 1)) * XAF - XA


print(R := fsolve(fobj2, 0.1))


def f(X):
  return 1 / (k * CA0 * CA0 * (1 - X) * X)


print(FA0 := 100 / ((R + 1) * quad(f, (R / (R + 1)) * XAF, XAF)))

print(Fc := FA0 * XAF)

R = 0.9
FA0 = 100 / ((R + 1) * quad(f, (R / (R + 1)) * XAF, XAF)[0])
print(FA0)

