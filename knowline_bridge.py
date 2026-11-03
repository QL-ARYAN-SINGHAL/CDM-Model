"""KnowLine-4B intent-question routes. Verity inference is independent.

KnowLine-4B-Gen5.5 (Q5_K_M GGUF) runs on llama-server. Each line x question pair is rendered with KnowLine's published
prompt layout (knowline_src/, read twice) and answered from the next-token probabilities of its two option labels;
nothing is generated. P(true) of the question is the score.
"""
import csv
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent
router = APIRouter(prefix="/knowline")
KNOWLINE_URL = os.environ.get("KNOWLINE_URL", f"http://127.0.0.1:{os.environ.get('KNOWLINE_PORT', '8009')}").rstrip("/")
KNOWLINE_MODEL_PATH = Path(os.environ.get(
    "KNOWLINE_MODEL_PATH", str(ROOT / "models" / "knowline" / "KnowLine-4B-Gen5.5.Q5_K_M.gguf")))
# Serving strategy, from benchmark_knowline.py runs: the whole message is judged once per intent (one model pass per intent,
# so the time does not grow with the number of lines). The input is read once (99.8% on 500 short prompts, ~0.4 s; 96.5% on
# 40 big prompts with 5 intents, ~5 s). KNOWLINE_BIG_READ_TWICE=1 reads prompts of BIG_WORDS words or more twice: 98.5%, ~10 s.
READ_TWICE = os.environ.get("KNOWLINE_READ_TWICE", "0") == "1"       # short prompts: KnowLine Gen5.5 read-twice layout
USE_CACHE = os.environ.get("KNOWLINE_CACHE", "0") == "1"             # llama-server prompt cache (measured: no gain on this model)
BIG_WORDS = int(os.environ.get("KNOWLINE_BIG_WORDS", "150"))         # prompts this long use BIG_READ_TWICE
BIG_READ_TWICE = os.environ.get("KNOWLINE_BIG_READ_TWICE", "0") == "1"  # 1: read long prompts twice (+2 points, about 2x the time)
_lock = threading.Lock()
tracking_lock = threading.Lock()
TRACKING_CSV = ROOT / "knowline_tracking.csv"
TRACKING_COLUMNS = ["tracking_id", "timestamp_utc", "model", "raw_prompt", "questions",
                    "selected_intent", "status", "intent_scores", "lines", "questions_asked",
                    "latency_ms", "low_threshold", "high_threshold"]
_scorer = None
_scorer_lock = threading.Lock()


def append_tracking(req, result):
    tracking_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    row = [tracking_id, timestamp, result["model"], req.raw_prompt,
           json.dumps([question.model_dump() for question in req.questions], ensure_ascii=False),
           result["selected_intent"] or "", result["status"],
           json.dumps(result["intent_scores"], ensure_ascii=False),
           json.dumps(result["lines"], ensure_ascii=False), result["questions_asked"],
           result["latency_ms"], req.low, req.high]
    with tracking_lock:
        new_file = not TRACKING_CSV.exists() or TRACKING_CSV.stat().st_size == 0
        with TRACKING_CSV.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(TRACKING_COLUMNS)
            writer.writerow(row)
    return tracking_id


@router.get("/tracking.csv")
def tracking_csv():
    with tracking_lock:
        if not TRACKING_CSV.exists():
            with TRACKING_CSV.open("w", encoding="utf-8", newline="") as file:
                csv.writer(file).writerow(TRACKING_COLUMNS)
    return FileResponse(TRACKING_CSV, media_type="text/csv", filename=TRACKING_CSV.name)


class Question(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=600)
    do: str = Field(min_length=1, max_length=600)
    dont: str = Field(min_length=1, max_length=600)


class ScanRequest(BaseModel):
    raw_prompt: str = Field(min_length=1, max_length=10000)
    questions: list[Question] = Field(min_length=1, max_length=16)
    low: float = Field(default=0.4, ge=0, le=1)
    high: float = Field(default=0.7, ge=0, le=1)

    @model_validator(mode="after")
    def validate_policy(self):
        if self.low >= self.high:
            raise ValueError("low must be below high")
        if len({q.name.casefold() for q in self.questions}) != len(self.questions):
            raise ValueError("question names must be unique")
        return self


def _request(path: str, body: dict | None = None, timeout: float = 10) -> dict:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        KNOWLINE_URL + path, data=payload,
        headers={"Content-Type": "application/json"} if payload else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(503, f"KnowLine server unavailable: {exc}") from exc


def _loaded_model() -> str:
    models = _request("/v1/models").get("data", [])
    expected = KNOWLINE_MODEL_PATH.name
    for item in models:
        model_id = item.get("id", "")
        if Path(model_id).name == expected:
            return model_id
    raise HTTPException(503, f"KnowLine-4B checkpoint not loaded at {KNOWLINE_URL}")


class _Scorer:
    """KnowLine's prompt layout (read twice) plus the label-token readout, on top of llama-server /completion."""

    def __init__(self, read_twice: bool = READ_TWICE, cache: bool = USE_CACHE):
        sys.path.insert(0, str(ROOT / "knowline_src"))
        from transformers import AutoTokenizer
        import knowline_server as ks
        from knowline_layout import Layout, render_request
        self._render_request = render_request
        self.cache = cache
        self.default_twice = read_twice
        self.layouts = {False: Layout(read_twice=False), True: Layout(read_twice=True)}
        self.tok = AutoTokenizer.from_pretrained(str(ROOT / "knowline_src"))
        _, probe, _ = ks.render(self.tok, "x", {"q": {"type": "noul"}}, ["A", "B"])
        self.labels, self.label_ids = ks.find_labels(self.tok, probe["q"][0])

    def p_true(self, state, question: dict, read_twice: bool | None = None) -> float:
        twice = self.default_twice if read_twice is None else read_twice
        rendered = self._render_request({"state": state, "questions": {"q": question}}, self.tok, self.layouts[twice],
                                        labels=self.labels)
        text, keys = rendered["prompts"]["q"][0]
        token_ids = self.tok.encode(text, add_special_tokens=False)
        wanted = self.label_ids[:len(keys)]
        data = _request("/completion", {"prompt": token_ids, "n_predict": 1, "temperature": 0, "n_probs": 200,
                                        "cache_prompt": self.cache}, timeout=180)
        try:
            top = data["completion_probabilities"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(502, "KnowLine response has no token probabilities") from exc
        found = {entry["id"]: entry["logprob"] for entry in top}
        logps = [found.get(i, -math.inf) for i in wanted]
        peak = max(logps)
        if not math.isfinite(peak):
            raise HTTPException(502, "KnowLine returned no probability for its answer labels")
        weights = [math.exp(x - peak) for x in logps]
        return dict(zip(keys, (w / sum(weights) for w in weights)))["true"]


def _get_scorer() -> _Scorer:
    global _scorer
    with _scorer_lock:
        if _scorer is None:
            _scorer = _Scorer()
        return _scorer


def choose_read_twice(raw_prompt: str) -> bool:
    """Read the input twice (more accurate, about twice the time) for long prompts only."""
    return BIG_READ_TWICE if len(raw_prompt.split()) >= BIG_WORDS else READ_TWICE


def compute_scan(scorer, raw_prompt, questions, low, high, read_twice=None):
    """Judge the whole message once per question -> (one result entry, best probability per intent).

    `questions` need .name/.description/.do/.dont. read_twice=None uses the scorer's default."""
    text = raw_prompt.strip()
    scores = []
    for question in questions:
        probability = round(scorer.p_true(text, {
            "type": "noul", "instructions": question.description,
            "criteria": {"true": question.do, "false": question.dont},
        }, read_twice=read_twice), 4)
        scores.append({"intent": question.name, "probability": probability,
                       "status": "match" if probability >= high else "review" if probability >= low else "no_match"})
    winner = max(scores, key=lambda score: score["probability"])
    entry = {"line": text, "scores": scores,
             "selected_intent": winner["intent"] if winner["status"] == "match" else None, "status": winner["status"]}
    return [entry], {score["intent"]: score["probability"] for score in scores}


@router.get("/health")
def health():
    try:
        model_id = _loaded_model()
        return {"ready": True, "model": model_id}
    except HTTPException as exc:
        return {"ready": False, "error": exc.detail}


@router.post("/scan")
def scan(req: ScanRequest):
    model_id = _loaded_model()
    if not req.raw_prompt.strip():
        raise HTTPException(422, "raw_prompt must contain text")
    twice = choose_read_twice(req.raw_prompt)
    scorer = _get_scorer()
    started = time.perf_counter()
    with _lock:
        line_results, max_by_intent = compute_scan(scorer, req.raw_prompt, req.questions, req.low, req.high,
                                                   read_twice=twice)
    ranked = sorted(max_by_intent.items(), key=lambda pair: pair[1], reverse=True)
    best_name, best_probability = ranked[0]
    result = {
        "model": model_id,
        "selected_intent": best_name if best_probability >= req.high else None,
        "status": "match" if best_probability >= req.high else
                  "review" if best_probability >= req.low else "no_match",
        "intent_scores": max_by_intent,
        "lines": line_results,
        "questions_asked": len(req.questions),
        "scope": "message",
        "read_twice": twice,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "thresholds": {"low": req.low, "high": req.high},
    }
    result["tracking_id"] = append_tracking(req, result)
    result["saved_to_tracker"] = True
    return result
