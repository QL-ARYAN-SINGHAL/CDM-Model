# CDM-Model

Portable local POC for testing two intent systems:

- **Fine-tuned Verity V4**: intent classification plus entity extraction.
- **Kev 0.8B**: dynamic typed intent questions with description, Do, Don't, raw prompt, thresholds, and per-line scores.

## Run locally

Requirements: macOS/Linux, Python 3.11+, `llama-server`, and Git LFS for model checkout.

```bash
git lfs install
git lfs pull
python3 -m venv venv
venv/bin/pip install -r requirements.txt
./run.sh
```

Open `http://127.0.0.1:8000/`.

`run.sh` starts Verity on port 8000 and Kev on port 8009. `start_public_test.sh` additionally starts the protected public proxy and temporary Cloudflare Quick Tunnel. The public proxy exposes inference only; tracker files remain local.

## Logs and exports

- `http://127.0.0.1:8000/tracker` shows separate Verity and Kev tabs.
- `/tracking.csv` stores Verity rows.
- `/kev/tracking.csv` stores Kev rows.
- `/tracking.xlsx` downloads workbook with `Fine-tuned Verity` and `Kev 0.8B` sheets.

Runtime logs, environment files, access keys, virtualenvs, caches, and raw JSONL logs stay out of Git.

## Model files

Model files use Git LFS. Verity checkpoint is under `models/verity/best`; Kev checkpoint is under `models/kev`. You can override paths with `VERITY_MODEL_PATH` and `KEV_MODEL_PATH`.

Kev is a Q8 GGUF checkpoint and requires `llama-server`. Its decision quality is not guaranteed for every domain; calibrate thresholds on reviewed positive, negative, and near-miss prompts.
