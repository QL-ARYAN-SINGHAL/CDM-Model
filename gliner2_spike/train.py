"""Step 2: fine-tune GLiNER2 (base model: entities + 3 classification tasks) with live progress and ETA. Single process or `torchrun --nproc_per_node=N train.py ...` (DDP).
Progress lines go to stdout and to <out>/progress.log; <out>/progress.json always holds the latest numbers.
usage: python train.py --work DIR --out DIR --model fastino/gliner2-base-v1 --epochs 3 --batch 8"""
import argparse, json, os, sys, time, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
try:
    import truststore; truststore.inject_into_ssl()
except Exception: pass
import torch
from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default="fastino/gliner2-base-v1")
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--encoder_lr", type=float, default=1e-5); ap.add_argument("--task_lr", type=float, default=5e-4); ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--log_every", type=int, default=25); ap.add_argument("--eval_every", type=int, default=0, help="0 = twice per epoch"); ap.add_argument("--max_steps", type=int, default=-1)
ap.add_argument("--workers", type=int, default=2); ap.add_argument("--lora", action="store_true"); ap.add_argument("--fp32", action="store_true")
A = ap.parse_args()
rank = int(os.environ.get("LOCAL_RANK", -1)); world = int(os.environ.get("WORLD_SIZE", 1))
os.makedirs(A.out, exist_ok=True); PLOG = os.path.join(A.out, "progress.log")
n_train = sum(1 for _ in open(os.path.join(A.work, "train.jsonl")))
steps_per_epoch = max(1, n_train // (A.batch * world * A.accum)); eval_every = A.eval_every or max(50, steps_per_epoch // 2)


def say(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"; print(line, flush=True)
    with open(PLOG, "a") as f: f.write(line + "\n")


class ETATrainer(ExtractorTrainer):
    def _log_metrics(self, metrics, prefix=""):
        super()._log_metrics(metrics, prefix)
        if prefix != "train" or not self.is_main_process or self.progress_bar is None: return
        m = metrics if isinstance(metrics, dict) else metrics.to_dict(); step = int(m.get("step", self.global_step)); total = int(self.progress_bar.total or 0)
        if not step or not total: return
        thr = float(m.get("throughput", 0)) or 1e-9; elapsed = float(m.get("samples_seen", 0)) / thr; eta = elapsed / step * max(0, total - step)
        gpu = " ".join(f"gpu{i}:{torch.cuda.max_memory_allocated(i) / 2**30:.1f}G" for i in range(torch.cuda.device_count())) if torch.cuda.is_available() else "cpu"
        fin = (datetime.datetime.utcnow() + datetime.timedelta(seconds=eta)).strftime("%H:%M UTC")
        say(f"step {step}/{total} ({100 * step / total:.1f}%) epoch {m.get('epoch', 0):.2f}/{A.epochs} loss {m.get('loss', float('nan')):.4f} lr {m.get('learning_rate', 0):.2e} | "
            f"{thr:.1f} samples/s | elapsed {common.fmt_eta(elapsed)} | ETA {common.fmt_eta(eta)} (done ~{fin}) | {gpu}")
        json.dump({"step": step, "total_steps": total, "percent": round(100 * step / total, 2), "loss": m.get("loss"), "elapsed_s": int(elapsed), "eta_s": int(eta), "finish_utc": fin, "samples_per_s": round(thr, 1)}, open(os.path.join(A.out, "progress.json"), "w"))

    def _evaluate(self, eval_dataset):
        t = time.time(); r = super()._evaluate(eval_dataset)
        if self.is_main_process: say(f"EVAL at step {self.global_step}: " + ", ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float)) + f" ({time.time() - t:.0f}s)")
        return r


if rank <= 0:
    say(f"world={world} gpus={torch.cuda.device_count()} train_rows={n_train} steps/epoch={steps_per_epoch} epochs={A.epochs} -> {steps_per_epoch * A.epochs} steps | effective batch {A.batch * world * A.accum} | eval every {eval_every} steps")
model = AutoExtractor.from_pretrained(A.model)
cfg = TrainingConfig(output_dir=A.out, experiment_name="gliner2_dual", num_epochs=A.epochs, max_steps=A.max_steps, batch_size=A.batch, eval_batch_size=A.batch * 2, gradient_accumulation_steps=A.accum,
                     encoder_lr=A.encoder_lr, task_lr=A.task_lr, scheduler_type="cosine", warmup_ratio=A.warmup, fp16=not A.fp32, bf16=False, eval_strategy="steps", eval_steps=eval_every,
                     save_total_limit=2, save_best=True, metric_for_best="eval_loss", greater_is_better=False, logging_steps=A.log_every, num_workers=A.workers, use_lora=A.lora,
                     local_rank=rank, validate_data=False, report_to_wandb=False, seed=42)
trainer = ETATrainer(model, cfg); t0 = time.time()
res = trainer.train(train_data=os.path.join(A.work, "train.jsonl"), eval_data=os.path.join(A.work, "val.jsonl"))
if rank <= 0:
    say(f"TRAINING DONE in {common.fmt_eta(time.time() - t0)} | steps {res['total_steps']} | best eval loss {res['best_metric']:.4f} | {res['samples_per_second']:.1f} samples/s")
    json.dump({k: v for k, v in res.items() if not k.endswith("history")} | {"eval_history": res.get("eval_metrics_history", [])}, open(os.path.join(A.out, "train_result.json"), "w"), indent=1, default=str)
