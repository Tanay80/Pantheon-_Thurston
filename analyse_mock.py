import numpy as np
for g in ["rh2s2", "nil", "solv"]:
    try:
        real = np.atleast_2d(np.loadtxt(f"{g}/res_{g}_pplus_real.txt"))[0]
        r = np.atleast_2d(np.loadtxt(f"{g}/res_{g}_pplus_mock.txt"))
    except OSError:
        continue
    if r.size == 0:
        print(f"{g}: no mock results yet"); continue
    d = r[:, 0]
    p = (1 + (d >= real[0]).sum()) / (1 + len(d))
    print(f"{g}: n={len(d)} real dchi2={real[0]:.2f} | mock median={np.median(d):.2f} "
          f"95%={np.percentile(d, 95):.2f} p={p:.3f} | frac(e2>0.19)={(r[:,2] > 0.19).mean():.2f} "
          f"| sigma mean={r[:,1].mean():.4f} std={r[:,1].std():.4f} | Ok 95%={np.percentile(r[:,4], 95):.3f}")
