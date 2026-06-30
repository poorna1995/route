# Experiments layout

Binary **LLM routing** study (select \(M_{\mathrm{lo}}\) or \(M_{\mathrm{hi}}\)); oracle \(r(q)\) = appropriate-model label. Voice: [`../research/program.md`](../research/program.md) §0.7.

## Methodology (four parts)

```text
Part I    Experimental Design     M1 prepare → M2 oracle+scorecard+select → M3 eval
Part II   Development             oracle → φ/ψ/χ extraction (Stages 4–5; 6–8 TBD)
Part III  Deployment              runtime: signals → x(q) → score → π → model  (route-demo)
Part IV   Evaluation              frozen π on R_t → accuracy, cost, Pareto (H4)
```

## CLI

```bash
python run.py all --setting experiments/candidates/arc.yaml --name arc-smoke --smoke

RUN=experiments/runs/<run_id>
python run.py prepare --run $RUN
python run.py oracle  --run $RUN
python run.py eval --run $RUN
python run.py model-independent --run $RUN
python run.py model-dependent --run $RUN --role M_lo
python run.py cross-model --run $RUN
python run.py develop --run $RUN
python run.py signal-validation --run $RUN
python run.py evaluate --run $RUN
python run.py route-demo --run $RUN --query-id <qid>
```

Stages 6–8 (signal validation, feature selection, policy calibration) are **removed from code** pending redesign. Existing run artifacts under `signals/analysis/` and `routing/policy.json` still work with `deploy/` and `evaluate`.

## Code layout

```
llm_routing/
  signal_schema.py       # φ/ψ/χ column names (was signal_validation/registry)
  develop.py             # Part II Stages 4–5
  deploy/                # Part III + optional deploy/train.py
  evaluate.py            # Part IV
  signals/phi|psi|chi/
```
