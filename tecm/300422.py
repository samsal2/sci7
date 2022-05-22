import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt


E_std_c = -((4 * (-157.30) - (2 * -237.178)) * 1000) / (4 * 96500)
E_std_a = -(-(4184 * -20.3)) / (2 * 96500)

E_c = E_std_c + 0.059 / 4 * 28
E_a = E_std_a + 0.059 / 2 * np.log(0.7 / 1) / np.log(10)

print(f"E˚\t{E_std_c},\t{E_std_a}")
print(f"E\t{E_c},\t{E_a}")

def log10(v):
  return np.log(v) / np.log(10)

b_c = -0.11
i0_c = 1e-7
b_a = 0.08
i0_a = 1e-5

# b = y - m * x
y0_c = E_c - b_c * log10(i0_c)

# b = y - m * x
y0_a = E_a - b_a * log10(i0_a)

l1 = Polynomial([y0_c, b_c])
l2 = Polynomial([y0_a, b_a])

i = np.linspace(log10(1e-7), log10(1e2))

print(f"c: {l1}")
print(f"a: {l2}")

# m1x + b1 = m2x + b2
# (m1 - m2)x = b2 - b1
# x = (b2 - b1) / (m1 - m2)

log10i_corr = (y0_a - y0_c) / (b_c - b_a)
i_corr = np.power(10.0, log10i_corr)
V_corr = l2(log10(i_corr))

#  cm | 1 mm |
#     | 10 cm  |

r = i_corr * (1 / 96500) * (1 / 2) * (55.845) / (1 / 7.875) * (10) * (3600) * (24) * (365)

print(f"log10(i_corr): {log10(i_corr)}")
print(f"i_corr: {i_corr}")
print(f"V_corr: {V_corr}")
print(f"r: {r}")

sebas = (E_c - E_a - log10(1e-7) * b_c + log10(1e-5) * b_a) / (b_a - b_c)
print(f"sebas: {sebas}")

plt.plot(i, l1(i))
plt.plot(i, l2(i))
# plt.ylim(y0_a, y0_c)
plt.grid()
plt.show()


