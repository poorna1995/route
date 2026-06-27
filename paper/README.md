# Paper

Compressed output of `research/`. Venue-specific files live here.

```
paper/
├── acl.tex              # main LaTeX draft
├── references.bib       # BibTeX
├── draft/               # §1–§8 LaTeX drafts (canonical prose; edit here)
├── tables/              # T1–T4 (.tex fragments)
└── figures/             # F1–F5 specs + PDFs (after experiments)
```

## Build (after installing acl.sty)

```bash
cd paper
pdflatex acl
bibtex acl
pdflatex acl
pdflatex acl
```

Until `acl.sty` is installed, `acl.tex` uses `geometry` + `natbib` as a fallback.

## Writing

| Location | Purpose |
| -------- | ------- |
| `draft/01--08_*.tex` | **Edit all sections here** (`\input` from `acl.tex`) |
| `tables/T1--T4` | Result tables (fill after GPU runs) |

**Locked narrative:** (1) pre-inference unsupervised signals → (2) three information sources (query-derived | model-derived | cross-model) → (3) characterization (I–III) → (4) routing evaluation (IV). C3 = layerwise extension within model-derived. SCOPE: contrast for timing/utility reporting only—not a template.

Outline: `research/11_paper_outline.md`
