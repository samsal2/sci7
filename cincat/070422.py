import numpy as np
import matplotlib.pyplot as plt


pbz_atm = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2])
cabs_umol_g_343k = np.array([220, 340, 680, 880])
cabs_umol_g_363k = np.array([45, 78, 170, 270, 420])
cabs_umol_g_403k = np.array([20, 39, 86, 160, 260])

def same_length(l1, l2):
   size = min(len(l1), len(l2)) 
   return np.resize(l1, (size,)), np.resize(l2, (size,))

def are(model, expected):
    return np.average(abs((model - expected) / expected) * 100)

def linearfit(X, Y):
    model = np.polynomial.Polynomial.fit(X, Y, deg=1).convert()
    err = are(model(X), Y)
    return model.coef, err

def find_ct_and_kads(p, cabs):
    p, cabs = same_length(p, cabs) 
    X = p
    Y = p / cabs
    coef, err = linearfit(X, Y)
    ct = 1 / coef[0]
    kads = 1 / (coef[1] * ct)
    return ct, kads, err

print(find_ct_and_kads(pbz_atm, cabs_umol_g_343k))
print(find_ct_and_kads(pbz_atm, cabs_umol_g_363k))
print(find_ct_and_kads(pbz_atm, cabs_umol_g_403k))
