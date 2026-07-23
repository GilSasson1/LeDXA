# Disease metadata

This directory contains small, participant-free mappings used to label and stratify disease
prediction results:

| File | Purpose |
|---|---|
| `disease_display_names.json` | Maps individual disease target IDs to publication labels. |
| `disease_groups.json` | Marks individual diseases as general, sex-specific, or directly defined by DXA measurements. |
| `disease_display_names_group.json` | Maps organ-system target IDs to publication labels. |
| `disease_groups_group.json` | Marks organ-system targets as general or sex-specific. |

These files are analysis metadata, not cohort data or result tables. Participant-level disease
targets belong under the git-ignored `data/` directory.
