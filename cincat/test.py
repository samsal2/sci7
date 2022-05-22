import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt


Ce = np.array([2.82, 4.77, 11.9, 15.93, 21.21, 26.48, 36.67, 44.31, 50.1])
qe = np.array([0.045, 0.053, 0.072, 0.079, 0.083, 0.09, 0.093, 0.098, 0.102])

X = np.concatenate(([np.ones(len(Ce))], [1 / Ce]), axis=0)
X = X.transpose()

Xa = X.transpose()
Y = 1 / qe

print(X)
print(Xa)

r = np.matmul(np.matmul(np.linalg.inv(np.matmul(Xa, X)), Xa), Y)

print(r)

qm = 1 / r[0]
print(f"qm {qm}")


# 2

V = 1000
C0 = 0.7
m_obj = 2
C = 0
K = 375
alfa = 0.7

tolerancia = 1000

while tolerancia > 0.001:
  m = V * (C0 - C) / (K * np.power(C, alfa))
  tolerancia = np.abs(m - m_obj)
  C = C + 0.0001

print(C)


# 3


x = np.linspace(-2 * np.pi, 2 * np.pi)
y_2_iteraciones = np.empty(len(x))
y_4_iteraciones = np.empty(len(x))
y_6_iteraciones = np.empty(len(x))
y_100_iteraciones = np.empty(len(x))

n = 2
for i in range(len(x)):
  s = 1

  for j in range(1, n):
    if j % 2 == 1:
      s = s - np.power(x[i], float(j) * 2) / np.math.factorial(float(j) * 2)
    else:
      s = s + np.power(x[i], float(j) * 2) / np.math.factorial(float(j) * 2)

  y_2_iteraciones[i] = s


n = 4
for i in range(len(x)):
  s = 1

  for j in range(1, n):
    if j % 2 == 1:
      s = s - np.power(x[i], float(j) * 2) / np.math.factorial(float(j) * 2)
    else:
      s = s + np.power(x[i], float(j) * 2) / np.math.factorial(float(j) * 2)

  y_4_iteraciones[i] = s


n = 6
for i in range(len(x)):
  s = 1

  for j in range(1, n):
    if j % 2 == 1:
      s = s - np.power(x[i], float(j) * 2) / np.math.factorial(float(j) * 2)
    else:
      s = s + np.power(x[i], float(j) * 2) / np.math.factorial(float(j) * 2)

  y_6_iteraciones[i] = s

print(np.math.factorial(100))

n = 100
for i in range(len(x)):
  s = 1

  for j in range(1, n):
    if j % 2 == 1:
      s = s - Decimal(np.power(x[i], j * 2)) / \
          Decimal(np.math.factorial(j * 2))
    else:
      s = s + Decimal(np.power(x[i], j * 2)) / \
          Decimal(np.math.factorial(j * 2))

  y_100_iteraciones[i] = s

plt.plot(x, y_2_iteraciones)
plt.plot(x, y_4_iteraciones)
plt.plot(x, y_6_iteraciones)
plt.plot(x, y_100_iteraciones)
plt.xlabel("$\Theta$")
plt.ylabel("$cos(\Theta)$")
plt.grid()
plt.legend()
plt.show()

print(y_2_iteraciones[0])

#
t = np.array([1, 10, 20, 30, 60, 90, 120, 150, 180, 190, 195,
              200, 202, 204, 206, 209, 210, 212, 214, 220, 230, 240])

cco = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.0185, 0.0585, 0.1252, 0.2525, 0.2887,
                0.3521, 0.4589, 0.5325, 0.5821, 0.6521, 0.7859, 0.8982, 0.9898, 1])


f = 1 - cco

s = 0

for i in range(len(t) - 1):
  s = s + (t[i + 1] - t[i]) * 0.5 * (f[i + 1] + f[i])


mads = 3000 * (200 / 1000) / 1000 * s
qsat = mads / 4


print(qsat)
