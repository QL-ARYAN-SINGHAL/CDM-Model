# Verity local tester

Run from any directory:

```bash
/Users/app/Downloads/results/run.sh
```

Open http://127.0.0.1:8000/ and refresh any older tester tab. Open http://127.0.0.1:8000/tracker to switch between fine-tuned Verity and Kev rows. `/tracking.xlsx` downloads one Excel workbook with separate `Fine-tuned Verity` and `Kev 0.8B` sheets. `/tracking.csv` and `/kev/tracking.csv` remain separate CSV downloads.

The **Kev 0.8B** tab uses the separate Q8_0 GGUF decision checkpoint and `llama-server` on `127.0.0.1:8009`. `run.sh` starts both services. Default checkpoint: `/Users/app/Desktop/office Projects/IntentGate/models/kev-0.8b/Kev-0.8B-Q8_0.gguf`; override with `KEV_MODEL_PATH` and `LLAMA_SERVER` if moved. `/kev/health` verifies the loaded checkpoint. Verity `/predict` and its model remain independent.

For Kev, add one or more named intent questions, each with description, **Do** (true criterion), and **Don't** (false criterion), then enter raw prompt. `/kev/scan` splits prompt by nonempty lines, sends structured `{full_message, lines}` plus every line × question as one Noul request, and returns all probabilities, per-line selection, and highest-scoring message intent. Successful scans append raw prompt, all question fields, thresholds, scores, per-line decisions, and selected intent to `kev_tracking.csv`. HTML tracker has separate model tabs; Excel workbook has separate model sheets and explicit Kev question columns. Review and match thresholds are test values; calibrate them on labeled prompts before enforcement. A mixed travel/card prompt currently gave Travel 85.8% and Banking 41.1%; Kev also selected Travel on card line, so line-level accuracy is unproven.

The startup script uses `/Users/app/Downloads/results/venv` and loads **Verity V4** from `/Users/app/Downloads/cdm-model-output-v4/verity_run/model/best`. Override the checkpoint with `VERITY_MODEL_PATH` when needed. New shared tests append to `verity_v4_tracking.csv`, including the model name. Historical trackers remain available.

Modes:

- **Classify:** return Verity intent scores. Threshold 0.5.
- **Extract:** return Verity entities only. Internal intent scores gate each entity family.
- **Both:** return both intent results and gated dual-task entities. Raw entity candidates are retained in the API response and result log for disagreement audits.

All modes use the same internal dual-task pass with the combined intent and entity schema used during training. Output selection does not change model context. A wrong family intent can still suppress valid extraction; inspect raw candidates in Both mode when investigating misses.

Single and batch views show Verity scores and entities. API `/health` reports model name, loaded checkpoint path, and status. Each successful `/predict` result is appended to `verity_v4_tracking.csv`. Optional JSONL logging uses `verity_v4_results.jsonl` when `VERITY_LOG_RESULTS=true`. CSV and JSON session downloads cover the current browser session.

No rule-based fallback or entity-derived intent is used. `None` means every scored intent fell below its model's threshold; inspect three scores before judging a miss. Extraction uses all 45 training labels at threshold 0.5. Current contract allows fiction under normal family rules and labelled CVV alone as PCI. Contract edits do not retrain checkpoints. Corrected 62-prompt comparison is in `contract_audit_2026-10-27/joint_current_policy/`; older reports used separate inference schemas or earlier policy.

## Shared Cloudflare test link

`./start_public_test.sh` starts the local Verity V4 API and Kev, then opens a temporary Cloudflare Quick Tunnel through `public_proxy.py` on port 8001. Anyone with the link can run Verity and Kev prompts; saved trackers, CSVs, and workbook remain available only at the local `http://127.0.0.1:8000/tracker`. The models remain on this computer; keep it awake, connected, and running while others test. The tunnel URL changes when restarted. Prompts remain limited to 10,000 characters each. Successful Verity and Kev prompts/results append to their respective CSV files on this computer; Verity batch CSV/JSON exports also stay in each tester's browser session.

The V4 shared CSV stores each successful prompt, mode, category, expected labels, intent scores, raw and gated entity predictions, latency, model name, and per-row comparison results. Add expectations before running:

- Intent labels: `PII; PHI; PCI`; use `NONE` for a reviewed negative, blank for unreviewed.
- Entity labels: one per line as `TYPE=value` for exact value checks, or `TYPE` for type-only checks; use `NONE` for no expected entities.
- Batch can use one shared expectation for newline-separated prompts, or TSV per row: `prompt<TAB>category<TAB>expected intents<TAB>expected entities`.

Use canonical entity names such as `EMAIL`, `MAC_ADDRESS`, and `HEALTH_PLAN_BENEFICIARY_NUMBER`. The `Download shared master CSV` button exports all testers' saved rows.

Use synthetic test prompts only. This is a temporary development tunnel, not a hosted production deployment. Stop `cloudflared` and the API process to take it offline. For a stable hostname, the Cloudflare account needs a domain and a named Tunnel; protect it with Cloudflare Access before wider or ongoing use.
