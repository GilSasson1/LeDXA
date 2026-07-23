# Disease metadata

This directory contains small, participant-free mappings used to label and stratify disease
prediction results, both read by `common/plot_style.py`:

| File | Purpose |
|---|---|
| `disease_display_names.json` | Maps individual disease target IDs to publication labels. |
| `disease_groups.json` | Marks individual diseases as general, sex-specific, or directly defined by DXA measurements. |

These files are analysis metadata, not cohort data or result tables. Participant-level disease
targets belong under the git-ignored `data/` directory.

(The organ-system-level counterparts of these two files were removed — unreferenced by any script,
and tied to the still-pending organ-system-grouping Supplementary Table; see
`supplement/SUPPLEMENT_MANIFEST.md` in the parent `DEXA/` project.)
