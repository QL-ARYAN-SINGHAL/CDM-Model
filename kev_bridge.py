"""Optional Kev-0.8B playground routes. Verity inference is independent."""
import os
import csv
import threading
import time
import urllib.error
import urllib.request
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

router = APIRouter(prefix="/kev")
KEV_URL = os.environ.get("KEV_URL", f"http://127.0.0.1:{os.environ.get('KEV_PORT', '8009')}").rstrip("/")
KEV_MODEL_PATH = Path(os.environ.get(
    "KEV_MODEL_PATH",
    str(Path(__file__).resolve().parent / "models" / "kev" / "Kev-0.8B-Q8_0.gguf"),
))
_lock = threading.Lock()
tracking_lock = threading.Lock()
TRACKING_CSV = Path(__file__).resolve().parent / "kev_tracking.csv"
TRACKING_COLUMNS = ["tracking_id", "timestamp_utc", "model", "raw_prompt", "questions",
                    "selected_intent", "status", "intent_scores", "lines", "questions_asked",
                    "latency_ms", "low_threshold", "high_threshold"]


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
        KEV_URL + path, data=payload,
        headers={"Content-Type": "application/json"} if payload else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(503, f"Kev server unavailable: {exc}") from exc


def _loaded_model() -> str:
    models = _request("/v1/models").get("data", [])
    expected = KEV_MODEL_PATH.name
    for item in models:
        model_id = item.get("id", "")
        if Path(model_id).name == expected:
            return model_id
    raise HTTPException(503, f"Kev-0.8B checkpoint not loaded at {KEV_URL}")


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
    lines = [line.strip() for line in req.raw_prompt.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(422, "raw_prompt must contain text")
    if len(lines) * len(req.questions) > 128:
        raise HTTPException(422, "limit is 128 line × question pairs")

    questions = {}
    for line_idx in range(len(lines)):
        for question_idx, question in enumerate(req.questions):
            questions[f"l{line_idx}__q{question_idx}"] = {
                "type": "noul",
                "instructions": (
                    f"Judge only lines[{line_idx}]. Use full_message only as context. "
                    f"{question.description}"
                ),
                "criteria": {"true": question.do, "false": question.dont},
            }
    started = time.perf_counter()
    with _lock:
        response = _request("/v1/systemone", {
            "model": model_id,
            "state": {"full_message": req.raw_prompt, "lines": lines},
            "questions": questions,
        }, timeout=180)
    answers = response.get("answers", {})
    line_results = []
    max_by_intent = {q.name: 0.0 for q in req.questions}
    for line_idx, line in enumerate(lines):
        scores = []
        for question_idx, question in enumerate(req.questions):
            key = f"l{line_idx}__q{question_idx}"
            try:
                probability = float(answers[key]["noul"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(502, f"Kev response missing valid {key}") from exc
            probability = round(probability, 4)
            max_by_intent[question.name] = max(max_by_intent[question.name], probability)
            scores.append({"intent": question.name, "probability": probability,
                           "status": "match" if probability >= req.high else
                           "review" if probability >= req.low else "no_match"})
        winner = max(scores, key=lambda score: score["probability"])
        line_results.append({"line": line, "scores": scores,
                             "selected_intent": winner["intent"] if winner["status"] == "match" else None,
                             "status": winner["status"]})
    ranked = sorted(max_by_intent.items(), key=lambda pair: pair[1], reverse=True)
    best_name, best_probability = ranked[0]
    result = {
        "model": model_id,
        "selected_intent": best_name if best_probability >= req.high else None,
        "status": "match" if best_probability >= req.high else
                  "review" if best_probability >= req.low else "no_match",
        "intent_scores": max_by_intent,
        "lines": line_results,
        "questions_asked": len(questions),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "thresholds": {"low": req.low, "high": req.high},
    }
    result["tracking_id"] = append_tracking(req, result)
    result["saved_to_tracker"] = True
    return result
