import numpy as np
import rh2s2_pplus as m
from scipy.optimize import minimize

def fast_ang_sep(ls, bs, la, ba):
    l, b = np.deg2rad(ls), np.deg2rad(bs)
    la_, ba_ = np.deg2rad(la), np.deg2rad(ba)
    c = np.sin(b)*np.sin(ba_) + np.cos(b)*np.cos(ba_)*np.cos(l - la_)
    return np.arccos(np.clip(c, -1.0, 1.0))
m.ang_sep = fast_ang_sep
m.ls = np.asarray(m.ls.deg); m.bs = np.asarray(m.bs.deg)

chi2 = lambda *a: -2*m.my_likelihood(*a, 0.)

# 1. LCDM limit at the Table 3 means, then fully optimized
print("LCDM at Table 3 means: %.2f" % chi2(0.734, 0., 0., 0.334, 0., 0., 0., -19.248))
r = minimize(lambda x: chi2(x[0], 0., 0., x[1], 0., 0., 0., x[2]),
             [0.734, 0.334, -19.248], method="Nelder-Mead",
             options=dict(xatol=1e-6, fatol=1e-5))
print("LCDM optimized: %.2f at h=%.4f Om=%.4f M0=%.4f" % (r.fun, *r.x))

# 2. The geometry best fit from the optimizer (rounded values)
best = np.array([0.724, -0.0092, 0.200, 0.266, 0.0197, 275., 7., -19.248])
print("geo at best fit: %.2f" % chi2(*best))

# 3. Is it smooth? Tiny random perturbations around the best fit
rng = np.random.default_rng(1)
scale = np.array([1e-3, 1e-4, 1e-3, 1e-3, 1e-4, 1., 1., 1e-3])
for i in range(6):
    p = best + scale*rng.standard_normal(8)
    p[2] = min(p[2], 0.2)
    print("perturbed %d: %.2f" % (i, chi2(*p)))
