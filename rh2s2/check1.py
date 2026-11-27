import time
import numpy as np
import rh2s2_pplus as m

# Test 1: does my fast numpy angle match astropy's?
a_astropy = m.ang_sep(m.ls, m.bs, 120., 30.)

def fast_ang_sep(ls, bs, la, ba):
    l, b = np.deg2rad(ls), np.deg2rad(bs)
    la_, ba_ = np.deg2rad(la), np.deg2rad(ba)
    c = np.sin(b)*np.sin(ba_) + np.cos(b)*np.cos(ba_)*np.cos(l - la_)
    return np.arccos(np.clip(c, -1.0, 1.0))

a_fast = fast_ang_sep(m.ls.deg, m.bs.deg, 120., 30.)
print("max angle difference (rad):", np.abs(a_astropy - a_fast).max())

# Test 2: likelihood at roughly the Table 3 means, with astropy
args = (0.732, -0.001, 0.1, 0.326, 0.001, 160., 0., -19.248, 0.)
t0 = time.time()
lnL1 = m.my_likelihood(*args)
t1 = time.time()
print("astropy version: lnL = %.4f, chi2 = %.2f, time = %.2f s" % (lnL1, -2*lnL1, t1 - t0))

# Test 3: same likelihood with the fast angle function
m.ang_sep = fast_ang_sep
m.ls = np.asarray(m.ls.deg)
m.bs = np.asarray(m.bs.deg)
t0 = time.time()
lnL2 = m.my_likelihood(*args)
t1 = time.time()
print("fast version:    lnL = %.4f, chi2 = %.2f, time = %.2f s" % (lnL2, -2*lnL2, t1 - t0))
print("lnL difference:", abs(lnL1 - lnL2))
