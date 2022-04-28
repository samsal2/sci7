import numpy as np
from scipy.stats import linregress
# TIPO PFR INTEGRAL

# -r_A = FA0 * dXA / dV = kCa^alfa
# dv = Adz
# -r_A = (FA0 / A) * (dXA / dz) = kCa^alfa
# -r_A = (FA0 / A) * (dXA / dz) = k(CA0 * (1 - XA))^alfa
# -r_A = (dXA / dz) = (A * k / FA0)(CA0 * (1 - XA))^alfa
# ln(dXA / dz)  =  ln((A * k * CA0^alfa) / FA0) + alfa * ln(1 - XA)


D_interno = 0.158  # cm

X_A = np.array([0.00, 1.93, 3.82, 5.68, 7.58, 9.25, 11.0]) / 100

intervalo = 5  # cm
z = np.arange(0, len(X_A)) * intervalo

X_A_p = np.gradient(X_A, intervalo, edge_order=2)
print(f"dX_A / dz: {X_A_p}")

X = 1 - X_A
Y = np.log(X_A_p)

result = linregress(X, Y)

alfa = result.slope
print(f"alfa: {alfa}")

intercept = result.intercept
print(f"intercept: {intercept}")

CA0 = 2.3256e-6
print(f"CA0: {CA0}")

A = np.pi * np.power(D_interno, 2) / 4
print(f"A: {A}")

FA0 = 4.558e-5
print(f"FA0: {FA0}")

# intercept = ln((A * k * CA0^alfa) / FA0)
k = np.exp(intercept) * FA0 / (A * np.power(CA0, alfa))
print(f"k: {k}")
