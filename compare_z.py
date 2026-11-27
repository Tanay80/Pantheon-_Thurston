import numpy as np
for g in ["rh2s2", "nil", "solv"]:
    m = np.loadtxt(f"{g}/res_{g}_pplus_mock.txt")[:, 0]
    for tag, f in [("zHD ", f"{g}/res_{g}_pplus_real.txt"),
                   ("zCMB", f"{g}/res_{g}_pplus_zcmb_real.txt")]:
        r = np.atleast_2d(np.loadtxt(f))[0]
        p = (1 + (m >= r[0]).sum()) / (1 + len(m))
        print(f"{g:6s} {tag} dchi2={r[0]:.2f} p={p:.3f} sigma={r[1]:.4f} "
              f"e2={r[2]:.3f} Ok={r[4]:.4f} l={r[5]:.0f} b={r[6]:.0f}")
