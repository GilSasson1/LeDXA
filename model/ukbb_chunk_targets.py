import sys, pandas as pd
k, n = int(sys.argv[1]), int(sys.argv[2])
d = pd.read_csv("ukbb_baseline_disease_targets.csv", index_col=0, nrows=5)
t = [c for c in d.columns if c.startswith("dis__")]
print(" ".join(t[k-1::n]))
