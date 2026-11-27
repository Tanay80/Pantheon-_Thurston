import glob, numpy as np

for f in sorted(glob.glob("rh2s2_chains/*.txt")):
    with open(f) as fh:
        header = fh.readline()
    if not header.startswith("#"):
        continue
    names = header[1:].split()
    cols = [i for i, n in enumerate(names) if n.startswith("chi2")]
    if not cols:
        continue
    d = np.loadtxt(f)
    d = np.atleast_2d(d)
    for i in cols:
        print(f, names[i], "min chi2 = %.2f" % d[:, i].min())
