#!/usr/bin/env python3
"""Verity local inference. Shared training schema with selectable output modes."""
import json
import csv
import html
from io import BytesIO
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openpyxl import Workbook
import uuid

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("VERITY_MODEL_PATH", str(ROOT / "models" / "verity" / "best")))
MODEL_NAME = os.environ.get("VERITY_MODEL_NAME", "Verity V4")
RESULTS_LOG = ROOT / "verity_v4_results.jsonl"
TRACKING_CSV = ROOT / "verity_v4_tracking.csv"
TRACKING_COLUMNS = ["tracking_id", "timestamp_utc", "session_id", "category", "mode", "prompt",
                    "expected_intents", "expected_entities", "predicted_intents", "intent_scores",
                    "raw_entities", "gated_entities", "intent_audit", "entity_audit", "latency_ms", "model"]
LOG_RESULTS = os.environ.get("VERITY_LOG_RESULTS", "false").lower() == "true"
sys.path.insert(0, str(ROOT / "gliner2_spike"))
import common

INTENT_THRESHOLD = 0.5  # Evaluation default; all three raw scores are returned.
EXTRACTION_THRESHOLD = 0.5  # Same threshold and all-label schema as training evaluation.
app = FastAPI(title="Verity")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
from knowline_bridge import TRACKING_CSV as KNOWLINE_TRACKING_CSV, TRACKING_COLUMNS as KNOWLINE_TRACKING_COLUMNS, tracking_lock as knowline_tracking_lock, router as knowline_router
app.include_router(knowline_router)
lock = threading.Lock()
tracking_lock = threading.Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    torch.set_num_threads(4)
model = None
load_error = None
try:
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(str(MODEL_PATH), map_location=device)
    model.eval()
except Exception as exc:
    load_error = str(exc)
    print(f"Model load failed: {load_error}", flush=True)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    mode: Literal["classify", "extract", "both"] = "both"
    category: Literal["positive", "negative", "near-miss", "messy"] | None = None
    session_id: str | None = Field(default=None, max_length=80)
    expected_intents: list[str] | None = None
    expected_entities: list[dict[str, str | None]] | None = None


def predict_one(text: str, mode: str) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {load_error}")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    started = time.perf_counter()
    # Schema is part of the model input. Keep it identical across output modes.
    with lock, torch.inference_mode():
        probabilities, spans = common.predict_joint(model, [text], batch_size=1, threshold=EXTRACTION_THRESHOLD)
    internal_scores = probabilities[0]
    family_flags = {f: internal_scores[f] >= INTENT_THRESHOLD for f in common.FAMILIES}
    raw_entities = [{"type": kind, "text": text[start:end], "score": round(float(conf), 4),
                     "start": start, "end": end} for kind, start, end, conf in spans[0]]
    entities = [e for e in raw_entities if common.family_open(e["type"], family_flags)] if mode in ("extract", "both") else []
    scores = internal_scores if mode in ("classify", "both") else None
    intents = [kind.upper() for kind in common.FAMILIES if scores is not None and scores[kind] >= INTENT_THRESHOLD]
    top = max(intents, key=lambda kind: scores[kind.lower()]) if intents else ("None" if scores is not None else None)
    gated_entities = [e for e in raw_entities if common.family_open(e["type"], family_flags)]
    return {"text": text, "mode": mode, "intent": top,
            "intent_score": round(scores[top.lower()], 6) if top not in (None, "None") else None,
            "intents": intents, "intent_scores": {k.upper(): float(v) for k, v in scores.items()} if scores is not None else None,
            "entities": entities,
            "raw_entities": raw_entities if mode == "both" else None,
            "inference_schema": "joint", "extraction_gated_by_intent": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "model": MODEL_NAME, "intent_threshold": INTENT_THRESHOLD,
            "extraction_threshold": EXTRACTION_THRESHOLD,
            "_audit_raw_entities": raw_entities,
            "_audit_gated_entities": gated_entities,
            "_audit_intents": [f.upper() for f in common.FAMILIES if internal_scores[f] >= INTENT_THRESHOLD],
            "_audit_intent_scores": {f.upper(): float(v) for f, v in internal_scores.items()}}


def normalize_value(value: str) -> str:
    return " ".join(value.casefold().split())


def build_audit(request: PredictionRequest, result: dict) -> dict:
    predicted_intents = set(result["_audit_intents"])
    allowed_intents = {f.upper() for f in common.FAMILIES} | {"NONE"}
    if request.expected_intents is not None and any(v.upper() not in allowed_intents for v in request.expected_intents):
        raise HTTPException(status_code=422, detail="expected_intents accepts PII, PHI, PCI, or NONE")
    expected_intents = None if request.expected_intents is None else {v.upper() for v in request.expected_intents if v.upper() != "NONE"}
    intent_audit = None if expected_intents is None else {
        "missed": sorted(expected_intents - predicted_intents),
        "extra": sorted(predicted_intents - expected_intents),
    }
    expected_entities = request.expected_entities
    entity_audit = None
    if expected_entities is not None:
        allowed_types = {v.upper() for v in common.ALL_TYPES} | {"NONE"}
        if any(e.get("type", "").upper() not in allowed_types for e in expected_entities):
            raise HTTPException(status_code=422, detail="expected_entities contains an unknown type")
        actual = result["_audit_raw_entities"]
        gold_types = [e["type"].upper() for e in expected_entities if e.get("type", "").upper() != "NONE"]
        pred_types = [e["type"].upper() for e in actual]
        gold_counts, pred_counts = {}, {}
        for typ in gold_types: gold_counts[typ] = gold_counts.get(typ, 0) + 1
        for typ in pred_types: pred_counts[typ] = pred_counts.get(typ, 0) + 1
        type_tp = sum(min(n, pred_counts.get(t, 0)) for t, n in gold_counts.items())
        type_fn = sum(gold_counts.values()) - type_tp
        type_fp = sum(pred_counts.values()) - type_tp
        gold_pairs = [(e["type"].upper(), normalize_value(e.get("text") or "")) for e in expected_entities if e.get("type", "").upper() != "NONE" and e.get("text")]
        pred_pairs = [(e["type"].upper(), normalize_value(e["text"])) for e in actual]
        values_complete = all(e.get("type", "").upper() == "NONE" or bool(e.get("text")) for e in expected_entities)
        exact_tp = sum(min(gold_pairs.count(x), pred_pairs.count(x)) for x in set(gold_pairs)) if values_complete else None
        entity_audit = {"type_tp": type_tp, "type_fp": type_fp, "type_fn": type_fn,
                        "exact_tp": exact_tp,
                        "exact_fp": len(pred_pairs) - exact_tp if values_complete else None,
                        "exact_fn": len(gold_pairs) - exact_tp if values_complete else None,
                        "expected_values_provided": len(gold_pairs)}
    return {"intent": intent_audit, "entities": entity_audit}


def append_tracking(request: PredictionRequest, result: dict, audit: dict, tracking_id: str, timestamp: str):
    columns = TRACKING_COLUMNS
    row = [tracking_id, timestamp, request.session_id or "", request.category or "", request.mode,
           request.text,
           "" if request.expected_intents is None else json.dumps(request.expected_intents, ensure_ascii=False),
           "" if request.expected_entities is None else json.dumps(request.expected_entities, ensure_ascii=False),
           json.dumps(result["_audit_intents"]),
           json.dumps(result["_audit_intent_scores"], ensure_ascii=False),
           json.dumps(result["_audit_raw_entities"], ensure_ascii=False),
           json.dumps(result["_audit_gated_entities"], ensure_ascii=False),
           json.dumps(audit["intent"], ensure_ascii=False), json.dumps(audit["entities"], ensure_ascii=False),
           result["latency_ms"], MODEL_NAME]
    with tracking_lock:
        new_file = not TRACKING_CSV.exists() or TRACKING_CSV.stat().st_size == 0
        with TRACKING_CSV.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if new_file: writer.writerow(columns)
            writer.writerow(row)


@app.post("/predict")
def predict(request: PredictionRequest, req: Request):
    result = predict_one(request.text, request.mode)
    tracking_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    audit = build_audit(request, result)
    append_tracking(request, result, audit, tracking_id, timestamp)
    result["tracking_id"] = tracking_id
    result["saved_to_master_csv"] = True
    result["audit"] = audit
    if LOG_RESULTS:
        with open(RESULTS_LOG, "a", encoding="utf-8") as file:
            file.write(json.dumps({"timestamp_utc": timestamp, **result}, ensure_ascii=False) + "\n")
    result.pop("_audit_raw_entities", None)
    result.pop("_audit_gated_entities", None)
    result.pop("_audit_intents", None)
    result.pop("_audit_intent_scores", None)
    return result


@app.get("/health")
def health():
    return {"status": "healthy" if model is not None else "unavailable",
            "model_loaded": model is not None, "model_path": str(MODEL_PATH), "model_name": MODEL_NAME,
            "device": device, "load_error": load_error,
            "inference_schema": "joint", "extraction_gated_by_intent": True,
            "intent_threshold": INTENT_THRESHOLD, "extraction_threshold": EXTRACTION_THRESHOLD}


@app.get("/tracking.csv")
def tracking_csv(req: Request):
    if not TRACKING_CSV.exists():
        with tracking_lock:
            if not TRACKING_CSV.exists():
                with TRACKING_CSV.open("w", newline="", encoding="utf-8") as file:
                    csv.writer(file).writerow(TRACKING_COLUMNS)
    return FileResponse(TRACKING_CSV, media_type="text/csv", filename=TRACKING_CSV.name)


def load_tracking_rows():
    with tracking_lock:
        if TRACKING_CSV.exists():
            with TRACKING_CSV.open(newline="", encoding="utf-8") as file:
                verity_records = list(csv.DictReader(file))
        else:
            verity_records = []
    with knowline_tracking_lock:
        if KNOWLINE_TRACKING_CSV.exists():
            with KNOWLINE_TRACKING_CSV.open(newline="", encoding="utf-8") as file:
                knowline_records = list(csv.DictReader(file))
        else:
            knowline_records = []
    return verity_records, knowline_records


def spreadsheet_text(value):
    value = str(value or "")
    return "'" + value if value.startswith(("=", "+", "-", "@")) else value


@app.get("/tracking.xlsx")
def tracking_workbook():
    verity_records, knowline_records = load_tracking_rows()
    workbook = Workbook(write_only=True)
    verity_sheet = workbook.create_sheet("Fine-tuned Verity")
    verity_sheet.append(TRACKING_COLUMNS)
    for record in verity_records:
        verity_sheet.append([spreadsheet_text(record.get(column)) for column in TRACKING_COLUMNS])
    knowline_columns = ["tracking_id", "timestamp_utc", "model", "raw_prompt", "intent_names",
                   "descriptions", "do_criteria", "dont_criteria", "questions_json",
                   "selected_intent", "status", "intent_scores", "lines", "questions_asked",
                   "latency_ms", "low_threshold", "high_threshold"]
    knowline_sheet = workbook.create_sheet("KnowLine 4B")
    knowline_sheet.append(knowline_columns)
    for record in knowline_records:
        try:
            questions = json.loads(record.get("questions") or "[]")
            if not isinstance(questions, list):
                questions = []
        except (ValueError, TypeError):
            questions = []
        values = dict(record)
        values["questions_json"] = record.get("questions") or ""
        for column, key in (("intent_names", "name"), ("descriptions", "description"),
                            ("do_criteria", "do"), ("dont_criteria", "dont")):
            values[column] = json.dumps([question.get(key, "") for question in questions if isinstance(question, dict)], ensure_ascii=False)
        knowline_sheet.append([spreadsheet_text(values.get(column)) for column in knowline_columns])
    output = BytesIO()
    workbook.save(output)
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": 'attachment; filename="model_tracking.xlsx"',
                                      "Cache-Control": "no-store"})


@app.get("/tracker", response_class=HTMLResponse)
def tracker():
    verity_records, knowline_records = load_tracking_rows()
    records = [(record.get("timestamp_utc") or "", "verity", record) for record in verity_records]
    records.extend((record.get("timestamp_utc") or "", "knowline", record) for record in knowline_records)
    records.sort(key=lambda item: item[0], reverse=True)
    rows = []
    for _, kind, record in records:
        prompt = html.escape(record.get("raw_prompt" if kind == "knowline" else "prompt") or "")
        columns = KNOWLINE_TRACKING_COLUMNS if kind == "knowline" else TRACKING_COLUMNS
        fields = "".join(
            f"<dt>{html.escape(column.replace('_', ' ').title())}</dt>"
            f"<dd>{html.escape(record.get(column) or '')}</dd>"
            for column in columns
        )
        if kind == "knowline":
            model_name = "KnowLine 4B"
            mode = "intent questions"
            category = ""
            decision = (record.get("selected_intent") or "None") + " · " + (record.get("status") or "")
        else:
            model_name = record.get("model") or ""
            mode = record.get("mode") or ""
            category = record.get("category") or ""
            decision = record.get("predicted_intents") or ""
        rows.append(
            f'<tr data-model="{kind}">'
            + f"<td>{html.escape((record.get('timestamp_utc') or '')[:19].replace('T', ' '))}</td>"
            + "".join(f"<td>{html.escape(value)}</td>"
                      for value in (model_name, mode, category))
            + f"<td><details class='prompt-details'><summary><span class='prompt-preview'>{prompt}</span>"
              "<span class='expand-label'>Show full prompt</span></summary>"
              f"<div class='prompt-full'>{prompt}</div></details></td>"
            + f"<td>{html.escape(decision)}</td>"
            + f"<td><details><summary>All fields</summary><dl>{fields}</dl></details></td></tr>"
        )
    page = (ROOT / "verity_tracker.html").read_text(encoding="utf-8")
    page = page.replace("<!--TRACKER_ROWS-->", "\n".join(rows))
    page = page.replace("<!--VERITY_COUNT-->", str(len(verity_records)))
    page = page.replace("<!--KNOWLINE_COUNT-->", str(len(knowline_records)))
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


@app.get("/")
def ui():
    return FileResponse(ROOT / "verity_ui.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
