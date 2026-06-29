# Scripts

## Environment setup

`pyproject.toml` is the source of truth. Optional dependency groups:

| Extra | Packages | When to use |
|-------|----------|-------------|
| *(core)* | datasets, pyyaml, huggingface_hub | prepare / corpus stages |
| `ml` | transformers, accelerate | RunPod (torch already in image) |
| `gpu` | torch | local dev without system torch |
| `dev` | pytest, numpy, scikit-learn | tests + geometry smoke |
| `semantic` | sentence-transformers, scikit-learn, numpy | Stage 5 real embeddings |

**RunPod (fast path):**

```bash
bash runpod.sh prefetch       # download both Llama weights first (recommended)
bash runpod.sh smoke          # provisions env + runs smoke test
bash runpod.sh pilot
bash runpod.sh resume experiments/runs/<id>
```

On first pod start, `scripts/setup_env.sh`:
- creates `/workspace/.venv-llm-routing` with `--system-site-packages` (inherits image PyTorch)
- installs only core + `[ml]` extras — no multi-GB torch download
- caches models under `/workspace/.cache/huggingface` (persists on network volume)

Subsequent runs skip install when `pyproject.toml` deps are unchanged (~seconds).

**Local dev:**

```bash
source scripts/setup_env.sh              # full install if torch missing
source scripts/setup_env.sh --force      # reinstall after dep changes
LLM_ROUTING_SKIP_ML=1 source scripts/setup_env.sh   # prepare-only, no torch
```

**Manual pip profiles:**

```bash
pip install -r requirements/base.txt   # core only
pip install -r requirements/ml.txt   # core + transformers (RunPod)
pip install -r requirements/full.txt   # core + torch + transformers
pip install -r requirements/dev.txt  # full + pytest
```

## Pipeline

```bash
python run.py new --setting experiments/candidates/arc.yaml --name arc-pilot
python run.py prepare --run experiments/runs/<run_id>
python run.py query-derived --run experiments/runs/<run_id> --mock-embed  # Stage 5: model-independent / H1
```

Partition IDs are frozen into the run's `setting.yaml` automatically.

## RunPod troubleshooting

**`Permission denied` / `401` / model download fails at `from_pretrained`:**

1. Create a [HF read token](https://huggingface.co/settings/tokens).
2. Accept model licenses: [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
3. On the pod:
   ```bash
   export HF_TOKEN=hf_...
   huggingface-cli login --token "$HF_TOKEN"
   ```
4. If download fails inside `xet_get` / `File reconstruction error`:
   ```bash
   export HF_HUB_DISABLE_XET=1
   export HF_HUB_ENABLE_HF_TRANSFER=0
   # remove corrupted partial 8B cache, then prefetch:
   rm -rf "$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct"
   bash runpod.sh prefetch
   python run.py resume --run experiments/runs/<run_id>
   ```

**Pipeline check without GPU weights:** `python run.py all --smoke --mock`
