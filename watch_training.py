"""
Live training monitor — refreshes every 30 seconds.

Usage:
    python watch_training.py                        # watch current Llama training
    python watch_training.py --log other.log        # watch a different log
    python watch_training.py --interval 10          # refresh every 10 seconds
"""

import argparse
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta

# Force UTF-8 output on Windows to avoid cp1252 codec errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


LOG_FILE = "training_llama32.log"
REFRESH  = 30           # seconds between refreshes
LOSS_HISTORY = 20       # number of loss values to plot in sparkline


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        name, util, mu, mt, temp, pwr, plim = r.stdout.strip().split(", ")
        vram_pct = float(mu) / float(mt) * 100
        return {
            "name": name.strip(),
            "util": int(util),
            "vram_used": int(mu),
            "vram_total": int(mt),
            "vram_pct": vram_pct,
            "temp": int(temp),
            "power": float(pwr),
            "power_limit": float(plim),
        }
    except Exception:
        return None


def parse_log(log_path):
    if not os.path.exists(log_path):
        return None

    with open(log_path, "rb") as f:
        content = f.read().decode("utf-8", errors="ignore")

    clean = lambda s: re.sub(r"\x1b\[[0-9;]*[mK]", "", s)
    parts = [clean(p.strip()) for p in re.split(r"[\r\n]+", content) if p.strip()]

    train_lines = [p for p in parts if "Training:" in p and "s/it" in p]
    info_lines  = [p for p in parts if "INFO" in p]

    result = {
        "epoch": 1,
        "total_epochs": 3,
        "done": 0,
        "total": 1,
        "elapsed": "0:00:00",
        "loss": None,
        "lr": None,
        "speed": None,
        "loss_history": [],
        "checkpoints": [],
        "val_losses": [],
        "start_time": None,
    }

    # Parse start time
    for l in info_lines[:5]:
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", l)
        if m:
            try:
                result["start_time"] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
            break

    # Parse epoch
    for l in info_lines:
        m = re.search(r"EPOCH (\d+)/(\d+)", l)
        if m:
            result["epoch"] = int(m.group(1))
            result["total_epochs"] = int(m.group(2))

    # Parse latest training batch
    if train_lines:
        last = train_lines[-1]
        m = re.search(r"\|\s*(\d+)/(\d+)\s*\[(\S+)<.*?,\s*([\d.]+)s/it.*?loss=([\d.]+).*?lr=([\S]+?)[\],]", last)
        if m:
            d, t, e, spd, loss, lr = m.groups()
            result.update({
                "done": int(d),
                "total": int(t),
                "elapsed": e,
                "loss": float(loss),
                "lr": lr,
                "speed": float(spd),
            })

    # Collect loss history from all training lines (sample every Nth)
    all_losses = []
    for l in train_lines:
        m = re.search(r"loss=([\d.]+)", l)
        if m:
            all_losses.append(float(m.group(1)))
    step = max(1, len(all_losses) // LOSS_HISTORY)
    result["loss_history"] = all_losses[::step][-LOSS_HISTORY:]

    # Validation losses and checkpoint events
    for l in info_lines:
        if "Val" in l and "Loss:" in l:
            m = re.search(r"Loss:\s*([\d.]+)", l)
            if m:
                result["val_losses"].append(float(m.group(1)))
        if "checkpoint" in l.lower() or "saved" in l.lower() or "best model" in l.lower():
            result["checkpoints"].append(l.split("INFO")[-1].strip().lstrip(" |-"))

    return result


def sparkline(values, width=30, min_val=None, max_val=None):
    """ASCII sparkline chart for loss trend."""
    bars = " .-:=+#@"
    if not values or len(values) < 2:
        return "  (not enough data)"

    lo = min_val if min_val is not None else min(values)
    hi = max_val if max_val is not None else max(values)
    span = hi - lo if hi != lo else 1.0

    line = ""
    for v in values[-width:]:
        idx = int((v - lo) / span * (len(bars) - 1))
        idx = max(0, min(len(bars) - 1, idx))
        line += bars[idx]
    return line


def bar(pct, width=30, fill="#", empty="-"):
    filled = int(pct / 100 * width)
    return fill * filled + empty * (width - filled)


def format_time(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def real_eta(done, total, elapsed_str):
    """Compute ETA from real average speed (not tqdm's jittery window)."""
    try:
        parts = elapsed_str.split(":")
        if len(parts) == 3:
            elapsed_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            elapsed_s = int(parts[0]) * 60 + float(parts[1])
        else:
            elapsed_s = float(parts[0])
        if done <= 0:
            return "calculating..."
        avg = elapsed_s / done
        remaining = (total - done) * avg
        return format_time(remaining)
    except Exception:
        return "unknown"


def render(log_path, refresh_interval):
    data = parse_log(log_path)
    gpu  = gpu_stats()
    now  = datetime.now().strftime("%H:%M:%S")

    clear()
    W = 60

    print("=" * W)
    print(f"  LLAMA 3.2-3B TRAINING MONITOR          [{now}]")
    print("=" * W)

    if data is None:
        print(f"\n  Log not found: {log_path}")
        print(f"  Retrying in {refresh_interval}s...")
        return

    # ── Epoch & progress ─────────────────────────────────────────────
    epoch_lbl = f"Epoch {data['epoch']} / {data['total_epochs']}"
    done, total = data["done"], data["total"]
    pct = done / total * 100 if total > 0 else 0
    eta = real_eta(done, total, data["elapsed"])

    print(f"\n  {epoch_lbl}")
    print(f"  [{bar(pct, 40)}] {pct:.1f}%")
    print(f"  Batches : {done:,} / {total:,}")
    print(f"  Elapsed : {data['elapsed']}")
    print(f"  ETA     : {eta}  (real avg, not tqdm estimate)")

    # ── Loss ────────────────────────────────────────────────────────
    print()
    if data["loss"] is not None:
        print(f"  Train Loss : {data['loss']:.4f}")
        print(f"  LR         : {data['lr']}")
    if data["val_losses"]:
        print(f"  Val Losses : {' -> '.join(f'{v:.4f}' for v in data['val_losses'])}")

    # Sparkline
    if data["loss_history"]:
        lo = min(data["loss_history"])
        hi = max(data["loss_history"])
        spark = sparkline(data["loss_history"])
        print(f"\n  Loss curve (high->low = good):")
        print(f"  {hi:.2f} {spark} {lo:.2f}")

    # ── GPU ──────────────────────────────────────────────────────────
    print()
    print(f"  {'-'*48}")
    if gpu:
        vram_bar = bar(gpu["vram_pct"], 20)
        util_bar = bar(gpu["util"], 20)
        print(f"  GPU         : {gpu['name']}")
        print(f"  Utilization : [{util_bar}] {gpu['util']}%")
        print(f"  VRAM        : [{vram_bar}] {gpu['vram_used']:,}/{gpu['vram_total']:,}MB  ({gpu['vram_pct']:.0f}%)")
        print(f"  Temperature : {gpu['temp']} C  (safe < 85 C)")
        print(f"  Power       : {gpu['power']:.0f}W / {gpu['power_limit']:.0f}W")
    else:
        print("  GPU stats unavailable")

    # ── Checkpoints ──────────────────────────────────────────────────
    print()
    print(f"  {'-'*48}")
    print(f"  Checkpoints saved:")
    if data["checkpoints"]:
        for c in data["checkpoints"][-5:]:
            print(f"    [OK] {c}")
    else:
        print("    (none yet -- saves at end of each epoch)")

    # ── Time estimate ────────────────────────────────────────────────
    print()
    print(f"  {'-'*48}")
    if data["start_time"] and done > 0 and total > 0:
        try:
            elapsed_parts = data["elapsed"].split(":")
            elapsed_s = int(elapsed_parts[0])*3600 + int(elapsed_parts[1])*60 + float(elapsed_parts[2])
            avg_s = elapsed_s / done
            remaining_ep1 = (total - done) * avg_s
            per_epoch_s   = (elapsed_s / (done / total))  # extrapolated full epoch
            total_remaining = remaining_ep1 + (data["total_epochs"] - data["epoch"]) * per_epoch_s
            finish_time = datetime.now() + timedelta(seconds=total_remaining)
            print(f"  Epoch {data['epoch']} finishes : {format_time(remaining_ep1)}")
            print(f"  All epochs finish : ~{format_time(total_remaining)}")
            print(f"  Est. completion   : {finish_time.strftime('%b %d %H:%M')}")
        except Exception:
            pass

    print()
    print(f"  Refreshing every {refresh_interval}s — Ctrl+C to quit")
    print("=" * W)


def main():
    parser = argparse.ArgumentParser(description="Live training monitor")
    parser.add_argument("--log", default=LOG_FILE, help="Log file path")
    parser.add_argument("--interval", type=int, default=REFRESH, help="Refresh seconds")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    if args.once:
        render(args.log, args.interval)
        return

    try:
        while True:
            render(args.log, args.interval)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
