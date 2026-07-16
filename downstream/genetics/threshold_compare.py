#!/usr/bin/env python3
"""Show how Fig-4c domain assignment depends on the LD-proxy r2 threshold.

Reads the cached fig4c_associations.tsv and reuses build_fig4c.DOMAIN_RULES.
Reports, for lead-only / r2>=0.95 / r2>=0.9 / r2>=0.8, the number of domains
per locus and the number of loci per domain -- so we can judge how much of the
pleiotropy is driven by large LD blocks sweeping in unrelated genes.
"""
import os
from downstream.genetics.build_fig4c import DOMAIN_RULES, LOCI, ASSOC_TSV

rows = []
with open(ASSOC_TSV) as fh:
    hdr = fh.readline().rstrip("\n").split("\t")
    for line in fh:
        rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))


def domains_for(trait):
    t = trait.lower()
    return [d for d, kws in DOMAIN_RULES.items() if any(k in t for k in kws)]


def assign(min_r2):
    loc_dom = {rsid: set() for rsid, _ in LOCI}
    for a in rows:
        try:
            r2 = float(a["r2"])
        except ValueError:
            continue
        if r2 < min_r2:
            continue
        for d in domains_for(a["trait"]):
            loc_dom[a["lead_snp"]].add(d)
    return loc_dom


THRESH = [("lead-only", 1.0), ("r2>=0.95", 0.95), ("r2>=0.9", 0.9), ("r2>=0.8", 0.8)]
assigns = {name: assign(r) for name, r in THRESH}

print("=== domains per locus (count) at each threshold ===")
print(f"{'gene':14s} " + " ".join(f"{n:>10s}" for n, _ in THRESH))
for rsid, gene in LOCI:
    print(f"{gene:14s} " + " ".join(
        f"{len(assigns[n][rsid]):>10d}" for n, _ in THRESH))

print("\n=== loci per domain at each threshold ===")
alld = sorted({d for a in assigns.values() for s in a.values() for d in s})
print(f"{'domain':26s} " + " ".join(f"{n:>10s}" for n, _ in THRESH))
for d in alld:
    print(f"{d:26s} " + " ".join(
        f"{sum(1 for s in assigns[n].values() if d in s):>10d}" for n, _ in THRESH))

print("\n=== RIN3 & PITPNM2 domains by threshold (the big LD blocks) ===")
for rsid, gene in [("rs17184313", "RIN3"), ("rs585522", "PITPNM2")]:
    for n, _ in THRESH:
        print(f"  {gene:9s} {n:10s}: {', '.join(sorted(assigns[n][rsid])) or '(none)'}")
