# Dataset experiment layout

The shared PERTINENCE method, paper assumptions, and terminology are summarized in
[`method.ko.md`](method.ko.md). Dataset-specific protocol and result claims belong to the
matching dataset directory and must not be promoted to project-wide claims.

Each dataset owns one self-contained directory:

```text
<dataset>/
├── configs/   # experiment and immutable asset manifests
├── data/      # versioned catalogs and compact, run-labelled result tables
├── docs/      # protocol, run-labelled results, and interpretation
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

Local and cloud execution records are kept apart: dataset directories own local experiment
protocol and results, while `../cloud/docs/runs/` owns GKE/KFP execution evidence.
