import glob, numpy as np
for g in ["lcdm", "rh2s2", "nil", "solv"]:
    for f in sorted(glob.glob(f"{g}/*chains/chain.1.txt")):
        with open(f) as fh:
            names = fh.readline()[1:].split()
        d = np.atleast_2d(np.loadtxt(f))
        w = d[:, names.index("weight")]
        print("\n", f, "rows =", len(d))
        for i, n in enumerate(names):
            if n in ("weight", "minuslogpost") or n.startswith("minuslogprior"):
                continue
            m = np.average(d[:, i], weights=w)
            s = np.sqrt(np.average((d[:, i] - m)**2, weights=w))
            extra = "  min = %.2f" % d[:, i].min() if n == "chi2" else ""
            print("  %-12s mean = %.4f  std = %.4f%s" % (n, m, s, extra))
    for s in glob.glob(f"{g}/*chains/**/*.stats", recursive=True):
        for line in open(s):
            if "log(Z)" in line:
                print("  ", s, line.strip())
