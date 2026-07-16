#!/usr/bin/env python3
"""Fail-closed guard against committing participant-level data.

Scans CSV/TSV headers for identifier columns and first-column values for UK Biobank
`eid` / HPP `10K_` id patterns, and flags binary data blobs. Prints only paths,
column names, and counts — never data values.

Usage:
    python tools/check_no_pii.py [paths...]        # default: whole repo
Exit status: 0 = clean, 1 = potential PII found (use in pre-commit / CI).
"""
import os
import re
import sys

TEXT_EXT = {".csv", ".tsv"}
BINARY_EXT = {".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5", ".pt", ".pth",
              ".ckpt", ".feather", ".parquet", ".safetensors"}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "wandb", "data"}

# Identifier column names (word-boundary matched; deliberately specific to avoid
# false positives like `pseudo_r2`).
ID_COL_RE = re.compile(
    r"(^eid$|^eid[_.]|registrationcode|sample_id|sampleid|subject_id|"
    r"participant_id|patient_id|date.?of.?birth|^dob$|birth.?year|year.?of.?birth)",
    re.I,
)
ID_VAL_RE = re.compile(r"^\s*(10K_\w+|\d{7})\b")


def scan_csv(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            header = f.readline()
            sep = "\t" if path.endswith(".tsv") else ","
            cols = [c.strip().strip('"') for c in header.split(sep)]
            id_cols = [c for c in cols if ID_COL_RE.search(c)]
            id_vals = any(ID_VAL_RE.match(line) for i, line in zip(range(200), f))
    except Exception:
        return None
    if id_cols or id_vals:
        return f"id_columns={id_cols or '-'} id_like_values={id_vals}"
    return None


def main(argv):
    roots = argv[1:] or ["."]
    hits = []
    for root in roots:
        if os.path.isfile(root):
            files = [root]
        else:
            files = []
            for dp, dns, fns in os.walk(root):
                dns[:] = [d for d in dns if d not in SKIP_DIRS]
                files += [os.path.join(dp, fn) for fn in fns]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in TEXT_EXT:
                info = scan_csv(f)
                if info:
                    hits.append((f, info))
            elif ext in BINARY_EXT:
                hits.append((f, "binary data blob (must not be committed)"))

    if hits:
        print(f"POTENTIAL PII — {len(hits)} file(s) flagged:")
        for f, info in hits:
            print(f"  {f}: {info}")
        print("\nRemove these or add to .gitignore before committing.")
        return 1
    print("check_no_pii: clean — no participant-level data detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
