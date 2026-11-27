import glob, numpy as np

for f in sorted(glob.glob("*/*chains*/*.txt")):
    with open(f) as fh:
        header = fh.readline()
    if not header.startswith("#"):
        continue
    names = header[1:].split()
    if "chi2" not in names:
        continue
    d = np.atleast_2d(np.loadtxt(f))
    print("%-40s min chi2 = %.2f" % (f, d[:, names.index("chi2")].min()))
