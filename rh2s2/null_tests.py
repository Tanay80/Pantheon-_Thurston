# usage (from inside the model folder):
#   python null_tests.py rh2s2_pplus real 1
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, importlib
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, os.getcwd())
MODNAME, MODE, NRUN = sys.argv[1], sys.argv[2], int(sys.argv[3])
NPROC = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()

H_FID, OM_FID, M0_FID = 0.734, 0.334, -19.248      # LCDM means, Table 3
NDIR = 20
OK_LO = -0.25 if "rh2s2" in MODNAME else 0.0
#              h     sigma  e2    Om    Ok     M0
LO   = np.array([0.5, -0.2, 0.0, 0.2, OK_LO, -21.0])
HI   = np.array([1.0,  0.2, 0.2, 0.5,  0.25, -18.0])
STEP = np.array([0.01, 0.005, 0.04, 0.03, 0.02, 0.03])

mod = importlib.import_module(MODNAME)

def fast_ang_sep(ls, bs, la, ba):
    l, b = np.deg2rad(ls), np.deg2rad(bs)
    la_, ba_ = np.deg2rad(la), np.deg2rad(ba)
    c = np.sin(b)*np.sin(ba_) + np.cos(b)*np.cos(ba_)*np.cos(l - la_)
    return np.arccos(np.clip(c, -1.0, 1.0))

mod.ang_sep = fast_ang_sep
mod.ls = np.asarray(mod.ls.deg, dtype=float)
mod.bs = np.asarray(mod.bs.deg, dtype=float)
LS0, BS0 = mod.ls.copy(), mod.bs.copy()
MASK = mod.mask
NSN = len(mod.z)

try:
    LCHOL = np.linalg.cholesky(mod.C)
except np.linalg.LinAlgError:
    w, V = np.linalg.eigh(mod.C)
    LCHOL = V * np.sqrt(np.clip(w, 0, None))

def lcdm_mu(h, Om):
    zg = np.linspace(0, mod.z.max(), 5000)
    cum = cumulative_trapezoid(1/np.sqrt(Om*(1+zg)**3 + 1 - Om), zg, initial=0)
    I = np.interp(mod.z, zg, cum)/h
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = 5*np.log10((1 + mod.z_hel)*I) + 42.384103515
    mu[~MASK] = mod.mu_host[~MASK]
    return mu

def make_mock(rng):
    return M0_FID + lcdm_mu(H_FID, OM_FID) + LCHOL @ rng.standard_normal(NSN)

def fit_lcdm():
    f = lambda x: -mod.my_likelihood(x[0], 0., 0., x[1], 0., 0., 0., x[2], 0.)
    r = minimize(f, [H_FID, OM_FID, M0_FID], method="Nelder-Mead",
                 bounds=[(LO[0], HI[0]), (LO[3], HI[3]), (LO[5], HI[5])],
                 options=dict(xatol=1e-5, fatol=1e-4, maxfev=800))
    return r.x, 2*r.fun

def nm(x0, la, ba, maxfev):
    f = lambda x: -mod.my_likelihood(x[0], x[1], x[2], x[3], x[4], la, ba, x[5], 0.)
    st = np.where(x0 + STEP > HI, -STEP, STEP)
    sim = np.vstack([x0] + [x0 + np.eye(6)[i]*st[i] for i in range(6)])
    return minimize(f, x0, method="Nelder-Mead", bounds=list(zip(LO, HI)),
                    options=dict(initial_simplex=sim, xatol=1e-5, fatol=1e-4, maxfev=maxfev))

def directions(n):
    k = np.arange(n)
    ba = np.degrees(np.arcsin((k + 0.5)/n))
    la = np.degrees(k*np.pi*(3 - np.sqrt(5))) % 360
    return la, ba

def analyse(seed):
    rng = np.random.default_rng(seed)
    if MODE == "mock":
        mod.m_b_corr = make_mock(rng)
    elif MODE == "scramble":
        idx = np.where(MASK)[0]
        p = rng.permutation(idx)
        mod.ls, mod.bs = LS0.copy(), BS0.copy()
        mod.ls[idx], mod.bs[idx] = LS0[p], BS0[p]
    xl, chi2_l = fit_lcdm()
    x0 = np.array([xl[0], 0.0, 0.1, xl[1], 0.01, xl[2]])
    las, bas = directions(NDIR)
    s1 = []
    for la, ba in zip(las, bas):
        r = nm(x0, la, ba, 250)
        s1.append((r.fun, r.x, la, ba))
    s1.sort(key=lambda t: t[0])
    best = None
    for fun, x, la, ba in s1[:2]:
        r = nm(x, la, ba, 1500)
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x, la, ba)
    fun, x, la, ba = best
    chi2_g = 2*fun
    return [chi2_l - chi2_g, x[1], x[2], x[3], x[4], la, ba, x[0], x[5], chi2_l, chi2_g]

if __name__ == "__main__":
    import time
    seeds = list(range(1000, 1000 + NRUN))
    out = f"res_{MODNAME}_{MODE}.txt"
    t0 = time.time()
    rows = []
    with open(out, "w") as fh:
        fh.write("# dchi2 sigma e2 Om Ok la ba h M0 chi2_lcdm chi2_geo\n")
        with Pool(min(NPROC, NRUN)) as pool:
            for r in pool.imap_unordered(analyse, seeds):
                rows.append(r)
                fh.write(" ".join(repr(float(v)) for v in r) + "\n")
                fh.flush()
                print("%d/%d done, %.1f min elapsed" % (len(rows), NRUN, (time.time()-t0)/60), flush=True)
    r = rows[0]
    print("dchi2 = %.2f | chi2_lcdm = %.2f | chi2_geo = %.2f" % (r[0], r[9], r[10]))
