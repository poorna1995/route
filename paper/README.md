# Paper — ACL v1

Maps to [`../research/program.md`](../research/program.md) and [`../research/nomenclature.md`](../research/nomenclature.md).

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Section map

| Paper section | File | Program / stage |
| ------------- | ---- | --------------- |
| Introduction | `sections/01_introduction.tex` | §0, §1 |
| Related Work | `sections/02_related_work.tex` | §2, literature |
| Problem Definition | `sections/03_problem_definition.tex` | §2–§3 |
| Method | `sections/04_*.tex` | §4–§8; Stages 5–8 |
| Experimental Setup | `sections/05_experimental_setup.tex` | §0.7, §13; Stages 1–4 |
| Results — H1–H3 | `sections/06a_results_signal_analysis.tex` | §5–§6, §10; Stage 6 |
| Results — H4 | `sections/06b_results_routing.tex` | §9, §10; Stage 9 |
| Discussion / Conclusion | `sections/07_*.tex`, `08_*.tex` | §14 |

Intro, Related Work, and Problem Definition are draft-complete (2026-06-28). Setup and results fill in after Stages 1–9.

**Paper voice (locked):** **binary LLM routing** / **model selection** — not *cascade paper*. Oracle \(r(q)\) = appropriate-model label. See [`../research/nomenclature.md`](../research/nomenclature.md) §1 and [`../research/program.md`](../research/program.md) §0.1a.

**Signal taxonomy (locked):** model-independent query-derived (H1) → model-dependent model-response (H2) → cross-model comparative (H3); H1–H3 = parallel tests of φ, ψ, χ **each alone**; φ+ψ combination = Stage 7 only.
