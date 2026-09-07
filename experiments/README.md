# Dataset experiment layout

Each dataset owns one self-contained directory:

```text
<dataset>/
├── configs/   # experiment and immutable asset manifests
├── data/      # versioned catalogs and compact result tables
├── docs/      # protocol, runbook, and result interpretation
├── figures/   # versioned plots
└── scripts/   # dataset-specific catalog and analysis code
```

Shared algorithm code belongs in `src/pertinence/`, and shared command-line
wrappers belong in the repository-level `scripts/` directory. Large downloaded
files, prediction caches, checkpoints, and raw run outputs belong in
`artifacts/<dataset>/` and are not committed.

Do not read another dataset's config, cache, or run directory from an
experiment-specific script. Cross-dataset comparisons should be implemented as
separate shared analysis code with all input paths supplied explicitly.
