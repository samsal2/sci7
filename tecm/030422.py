import numpy as np
import matplotlib.pyplot as plt

def log10(v):
  return np.log(v) / np.log(10)


E_std_zn = -0.763
E_std_h = 0

@np.vectorize
def f(zn, pH=3):
  return (E_std_h - E_std_zn) - 0.059 / 2 * log10(zn) - 0.059 * pH

print(f(1e-5))

zn = np.linspace(1e-6, 1e-1)
plt.plot(zn, f(zn))
plt.ylabel("$E$")
plt.xlabel("$[Zn^{2+}]$")
plt.show()

icorr  = 0.004347826087e3
