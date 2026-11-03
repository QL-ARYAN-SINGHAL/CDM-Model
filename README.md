# CDM-Model

Portable local POC for testing two intent systems:

- **Fine-tuned Verity V4**: intent classification plus entity extraction.
- **KnowLine 4B** (KnowLine-4B-Gen5.5, Q5_K_M GGUF): dynamic typed intent questions with description, Do, Don't, raw prompt, thresholds, and per-line scores.

## Run locally

Requirements: macOS/Linux, Python 3.11+, `llama-server` (llama.cpp), and Git LFS for the Verity checkpoint.

```bash
git lfs install
git lfs pull
python3 -m venv venv
venv/bin/pip install -r requirements.txt
./run.sh
```

Open `http://127.0.0.1:8000/`.

`run.sh` starts Verity on port 8000 and KnowLine on port 8009. `start_public_test.sh` additionally starts the protected public proxy and temporary Cloudflare Quick Tunnel (`PROXY_PORT` picks the proxy port, default 8001). The public proxy exposes inference only; tracker files remain local. `start_tracker.sh` serves a separate access-key protected live tracker (Verity and KnowLine tabs) on its own tunnel.

## Logs and exports

- `http://127.0.0.1:8000/tracker` shows separate Verity and KnowLine tabs.
- `/tracking.csv` stores Verity rows.
- `/knowline/tracking.csv` stores KnowLine rows.
- `/tracking.xlsx` downloads workbook with `Fine-tuned Verity` and `KnowLine 4B` sheets.

Runtime logs, environment files, access keys, virtualenvs, caches, and raw JSONL logs stay out of Git. `archive/` holds the earlier Kev 0.8B tracking rows.

## Model files

Verity checkpoint is under `models/verity/best` (Git LFS). KnowLine's GGUF is `models/knowline/KnowLine-4B-Gen5.5.Q5_K_M.gguf` (about 3.2 GB, not committed; download it from `mradermacher/KnowLine-4B-Gen5.5-GGUF`). Its prompt layout and tokenizer are vendored in `knowline_src/` (Apache-2.0, from `PelaAI/KnowLine-4B-Gen5.5`). Override paths with `VERITY_MODEL_PATH` and `KNOWLINE_MODEL_PATH`.

KnowLine runs on stock `llama-server`, but not through its `/v1/systemone` route: `knowline_bridge.py` renders KnowLine's own prompt and reads the probability of its two answer labels from `/completion`. Strategy (measured with `benchmark_knowline.py`; sets built by `build_benchmark_sets.py`):

- **The whole message is judged once per intent**, never line by line. Time therefore grows with the number of intents and the length of the text, not with the number of lines (line by line took 7-8 s on an 8-line prompt, about 0.7 s as a whole).
- **The input is read once**, for every length: 99.8% correct on 500 labelled short prompts (about 0.4 s per prompt) and 96.5% of intent decisions on 40 big prompts with 5 intents each (about 5 s; a 1,500-word text about 3.5 s).
- **Optional:** `KNOWLINE_BIG_READ_TWICE=1` reads prompts of 150 words or more twice (`KNOWLINE_BIG_WORDS`): 98.5% on the big prompts, but about twice the time.
- Switches: `KNOWLINE_READ_TWICE`, `KNOWLINE_BIG_WORDS`, `KNOWLINE_BIG_READ_TWICE`, `KNOWLINE_CACHE`. The result has one entry for the whole message instead of one per line.

Its decision quality is not guaranteed for every domain; calibrate thresholds on reviewed positive, negative, and near-miss prompts.
