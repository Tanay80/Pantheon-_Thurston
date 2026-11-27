import numpy as np

def wq(x, w, qs):
    """weighted quantiles; inf values sort to the end"""
    i = np.argsort(x)
    x, w = x[i], w[i]
    c = (np.cumsum(w) - 0.5*w) / w.sum()
    return [np.interp(q, c, np.where(np.isfinite(x), x, 1e12)) for q in qs]

for g in ["rh2s2", "nil", "solv"]:
    f = f"{g}/{g}_chains/chain.1.txt"
    with open(f) as fh:
        names = fh.readline()[1:].split()
    d = np.atleast_2d(np.loadtxt(f))
    w  = d[:, names.index("weight")]
    e2 = d[:, names.index("e2")]
    ok = d[:, names.index("om_k")]
    okp = np.abs(ok)

    a0 = (1 - e2)**(-1/6)
    b0 = (1 - e2)**(1/3)
    with np.errstate(divide="ignore", invalid="ignore"):
        if g == "rh2s2":
            ratio = 1/(a0*np.sqrt(3*okp));        ms = 1/np.sqrt(3*okp)
        elif g == "nil":
            ratio = 1/(2*b0*np.sqrt(3*okp));      ms = 1/(2*np.sqrt(1 - e2)*np.sqrt(3*okp))
        else:
            ratio = 1/(b0*np.sqrt(3*okp));        ms = 1/(np.sqrt(1 - e2)*np.sqrt(3*okp))

    print(f"\n{g}: {len(d)} samples")
    if g == "rh2s2":
        lo, hi = wq(ok, w, [0.025, 0.975])
        print("  Ok 95%% interval: [%.3f, %.3f]" % (lo, hi))
    else:
        print("  Ok 95%% upper limit: %.3f" % wq(ok, w, [0.95])[0])
    q = wq(ratio, w, [0.01, 0.05, 0.50])
    print("  L0/R0 (consistent a0,b0): 99%% lower = %.3f | 95%% lower = %.3f | median = %.2f" % tuple(q))
    qm = wq(ms, w, [0.01, 0.05, 0.50])
    print("  L0/R0 (manuscript formula): 99%% lower = %.3f | 95%% lower = %.3f | median = %.2f" % tuple(qm))
