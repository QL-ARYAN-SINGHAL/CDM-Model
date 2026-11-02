# Kev-0.8B Q8_0 live decision audit — 2026-10-30

## Setup

- Loaded model: `/Users/app/Desktop/office Projects/IntentGate/models/kev-0.8b/Kev-0.8B-Q8_0.gguf` through `llama-server` and the live Verity-side `/kev/scan` endpoint.
- Four fixed, independently scored intent questions: `personal`, `banking`, `api_key`, `gov_id`. Each carried one description plus explicit Do/Don't criteria; see `run.py`.
- 30 predeclared synthetic single-line prompts: 10 positives, 10 ordinary negatives, 10 near misses. Expected labels were set in `run.py` before inference. High threshold 0.7, review threshold 0.4 are UI test values, not calibrated policy.
- `results.csv` keeps every full prompt, expected/selected intent, all scores, per-line result, latency, and error. No API or regex fallback generated decisions.

## Results

| Category | Correct final decision | Other outcomes |
| --- | ---: | --- |
| Positive | 6/10 correct intent matches | 3 review, 1 no-match |
| Ordinary negative | 10/10 no-match | 0 review, 0 match |
| Near miss | 5/10 no-match | 4 review, 1 false match |

Decisive failures:

- `P05`: bank account plus IFSC scored Banking **0.3825**, below review threshold.
- `P02` phone and `P03` postal address scored Personal **0.4257** and **0.4415**; both only reached review.
- `P06` CVV scored Banking **0.5346**; review.
- `M08`: masked password `********` scored API key **0.8656**, a false match despite explicit Don't criterion excluding masked values.
- `M07`: `API_KEY` variable name scored API key **0.6953**, nearly matching. `M05` masked card scored Banking **0.5359**; review.

Reapplying different high thresholds to the *same saved model scores*:

| High threshold | Positives matched | Ordinary negatives falsely matched | Near misses falsely matched |
| ---: | ---: | ---: | ---: |
| 0.5 | 7/10 | 0/10 | 4/10 |
| 0.7 | 6/10 | 0/10 | 1/10 |
| 0.9 | 1/10 | 0/10 | 0/10 |

Two additional multiline probes (outside the 30) tested the linked approach's structured `{full_message, lines}` state and one Noul per line × intent. Both returned no selected intent despite visible values: `L01` bank account/IFSC reached Banking **0.4532**; `L02` email plus card reached Personal **0.6220** and Banking **0.5724**. The per-line scores in `multiline_results.json` also show scores bleeding between lines.

## Assessment

Kev-0.8B Q8_0 is **not suitable as the sole automatic allow/mask/block decision maker under this setup**. It misses visible sensitive values, flags a masked password, and fails these multiline probes. Keep it as a testing or review signal; do not treat a 0.7 probability as calibrated reliability. This small synthetic audit does not measure production accuracy or compare fairly against Verity. Next evaluation should use reviewed, production-shaped cases and tune thresholds on a separate calibration set.
