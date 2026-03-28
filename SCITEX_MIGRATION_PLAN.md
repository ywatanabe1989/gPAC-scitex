# SciTeX Writer Migration Plan

## Paper Overview

**Title:** gPAC: GPU-accelerated implementation of phase-amplitude coupling analysis

**Authors:** Yusuke Watanabe, Takufumi Yanagisawa

**Keywords:** phase-amplitude coupling, gpu, parallel computing

**Figures:** 8 main figures (Figure_ID_00 through Figure_ID_08, skipping 06), TEX captions in both `src/` and `compiled/`
**Tables:** 1 main table (Table_ID_01), plus 3 legacy tables in `.tex/.tmp/`
**Bibliography:** 15 lines in bibliography.bib (very sparse -- likely needs enrichment)

---

## Manuscript Completeness Assessment

| Section | Status | Notes |
|---|---|---|
| Title | Complete | "gPAC: GPU-accelerated implementation of phase-amplitude coupling analysis" |
| Abstract | Complete | Well-written, mentions 100x speedup, terabyte-scale processing |
| Introduction | Complete | Structured with [START/END] annotation tags (need cleanup) |
| Methods | Partial | Has `[fixme]` placeholders for physiological data source and PyTorch version |
| Results | Partial | Subsections exist but text is skeletal; figure/table refs use `\hlref{}` |
| Discussion | Incomplete | Only "Discussion here." placeholder |
| Bibliography | Sparse | Only 15 lines -- needs significant enrichment |
| Figures | Present | 8 figure TEX captions exist; need to verify image files |
| Tables | Present | 1 main table; 3 legacy tables in temp directory |
| Supplementary | Template | Minimal content, mostly placeholder structure |
| Revision | Present | Has reviewer1, reviewer2, and editor response structure |

---

## Current Structure -> SciTeX Writer Structure Mapping

### Shared Resources: `paper/manuscript/src/` -> `scitex-writer/00_shared/`

| Current File | SciTeX Target |
|---|---|
| `src/title.tex` | `00_shared/title.tex` |
| `src/authors.tex` | `00_shared/authors.tex` |
| `src/journal_name.tex` | `00_shared/journal_name.tex` |
| `src/keywords.tex` | `00_shared/keywords.tex` |
| `src/bibliography.bib` | `00_shared/bib_files/bibliography.bib` |
| `src/styles/` (columns, packages, formatting, bibliography, linker) | `00_shared/latex_styles/` |

### Manuscript Content: `paper/manuscript/src/` -> `scitex-writer/01_manuscript/contents/`

| Current File | SciTeX Target |
|---|---|
| `src/abstract.tex` | `01_manuscript/contents/abstract.tex` |
| `src/introduction.tex` | `01_manuscript/contents/introduction.tex` |
| `src/methods.tex` | `01_manuscript/contents/methods.tex` |
| `src/results.tex` | `01_manuscript/contents/results.tex` |
| `src/discussion.tex` | `01_manuscript/contents/discussion.tex` |
| `src/data_availability.tex` | `01_manuscript/contents/data_availability.tex` |
| `src/additional_info.tex` | `01_manuscript/contents/additional_info.tex` |
| `src/highlights.tex` | `01_manuscript/contents/highlights.tex` |
| `src/graphical_abstract.tex` | `01_manuscript/contents/graphical_abstract.tex` |
| `src/wordcount.tex` | `01_manuscript/contents/wordcount.tex` |
| `manuscript/main.tex` | `01_manuscript/manuscript.tex` (rewritten to use SciTeX paths) |

### Figures: `paper/manuscript/src/figures/` -> `scitex-writer/01_manuscript/contents/figures/`

| Current File | SciTeX Target |
|---|---|
| `figures/src/Figure_ID_00_pac_value_comparision.jpg.tex` .. `Figure_ID_08_n_perm.jpg.tex` | `contents/figures/caption_and_media/00_*.tex` .. `08_*.tex` |
| Figure image files (JPG) | `contents/figures/caption_and_media/jpg_for_compilation/` |
| `figures/.tex/.All_Figures.tex` | `contents/figures/compiled/FINAL.tex` |

### Tables: `paper/manuscript/src/tables/` -> `scitex-writer/01_manuscript/contents/tables/`

| Current File | SciTeX Target |
|---|---|
| `tables/src/Table_ID_01.tex` | `contents/tables/caption_and_media/01_*.tex` |
| Legacy tables in `.tex/.tmp/` | Review and migrate if needed |

### Supplementary: `paper/supplementary/` -> `scitex-writer/02_supplementary/`

| Current File | SciTeX Target |
|---|---|
| `supplementary/src/methods.tex` | `02_supplementary/contents/methods.tex` |
| `supplementary/src/results.tex` | `02_supplementary/contents/results.tex` |
| `supplementary/src/bibliography.bib` | Use shared bib from `00_shared/` |

### Revision: `paper/revision/` -> `scitex-writer/03_revision/`

The existing `revision/` directory has structured reviewer/editor responses that map into `03_revision/contents/`.

---

## What Needs to Change

### Phase 1: Content Migration (no rewriting)
1. Copy all section TEX content into SciTeX directory layout
2. Copy bibliography into `00_shared/bib_files/`
3. Copy figure image files; verify JPG availability for compilation
4. Copy table TEX files
5. Update `\input{}` paths in `manuscript.tex` to use SciTeX conventions
6. Adapt LaTeX style files to SciTeX `00_shared/latex_styles/`
7. Clean up `[START/END]` annotation tags from introduction.tex

### Phase 2: Content Completion
1. Fix `[fixme]` placeholders in methods.tex (physiological data source, PyTorch version)
2. Flesh out results.tex with actual quantitative results
3. Write discussion.tex (currently empty placeholder)
4. Enrich bibliography.bib (currently only 15 lines -- needs all cited references)
5. Verify all `\hlref{}` figure references resolve correctly

### Phase 3: Verification
1. Compile manuscript via `scitex writer compile` to confirm no LaTeX errors
2. Verify all figures render correctly
3. Verify bibliography resolves all citations
4. Compare compiled PDF against any existing compiled version

### Phase 4: Journal Adaptation
1. Update `journal_name.tex` for target journal
2. Adjust formatting/style files per journal guidelines
3. Use `scitex writer guideline-build` if journal template available

---

## Journal Candidates

The paper covers GPU-accelerated PAC analysis for neuroscience. Suitable targets:

| Journal | IF (~) | Fit | Notes |
|---|---|---|---|
| **Journal of Neuroscience Methods** | ~3.0 | High | Methods paper, perfect topical match |
| **NeuroImage** | ~5.7 | High | Strong on methods + neural data analysis |
| **Frontiers in Neuroinformatics** | ~3.5 | High | Neuroinformatics/tools focus |
| **PLOS Computational Biology** | ~4.3 | Medium-High | Computational neuroscience tools |
| **Journal of Open Source Software** | ~3.4 | High | Open-source software paper format |
| **SoftwareX** | ~3.4 | High | Software description paper |
| **GigaScience** | ~7.5 | Medium | Big data tools, open access |

---

## Experiment Code

The `scripts/` directory contains experiment code:
- `exp_01_synthetic_data_preparation/` -- Synthetic data generation
- `exp_02_tensorpac_comparison/` -- Comparison with TensorPAC
- `exp_03_advanced_statistics/` -- Statistical analyses
- `benchmarking/` -- Performance benchmarks
- `_learnability/` -- Trainable PAC experiments
- `_profile/` -- Profiling code

This code should be preserved as-is in the repo root alongside the `scitex-writer/` directory.

---

## Repo Structure After Migration

```
gPAC-scitex/
  paper/                 # Original (preserved for reference)
  scripts/               # Experiment code (preserved)
  data/                  # Data files
  config/                # Configuration
  tests/                 # Tests
  scitex-writer/         # SciTeX writer project
    00_shared/           # Shared: title, authors, bib, styles
    01_manuscript/       # Main manuscript content
    02_supplementary/    # Supplementary materials
    03_revision/         # Revision documents
    config/              # Compilation configs
    compile.sh           # One-command compilation
  SCITEX_MIGRATION_PLAN.md
```

---

## Status

- [x] GitHub repo created: https://github.com/ywatanabe1989/gPAC-scitex
- [x] All original content pushed (main + develop branches)
- [x] SciTeX writer template initialized at `scitex-writer/`
- [x] Migration plan written
- [ ] Content migration (Phase 1)
- [ ] Content completion (Phase 2)
- [ ] Compilation verification (Phase 3)
- [ ] Journal selection and adaptation (Phase 4)
