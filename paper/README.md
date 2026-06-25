# Paper

Compressed output of `research/`. Venue-specific files live here.

```
paper/
├── acl.tex              # main LaTeX draft
├── references.bib       # BibTeX
├── draft/               # §1–§8 LaTeX drafts (canonical prose; edit here)
├── sections/            # thin wrappers → draft/05–08 (legacy paths)
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

**Locked narrative:** (1) pre-hoc unsupervised routing signals → (2) two signal families → (3) characterization (I–II) → (4) complementary predictive information (III) → (5) decision utility (IV). SCOPE: contrast for timing/utility reporting only—not a template.

Outline: `research/11_paper_outline.md`
