"""
REAgents performance benchmark harness (T001 / plan P1.1).

Measures the hot-path latencies that Phase 3 optimizations target, so every
later change can be proven with numbers instead of vibes:

  - capture_fps       — sustained ScreenCapture.get_frame() rate
  - yolo_ms           — PerceptionSystem.detect_objects latency (p50/p95/mean)
  - ocr_ms            — read_hud OCR portion latency
  - health_ms         — get_health_percentage colour-mask latency
  - perception_ms     — full perception tick (YOLO + HUD + death screen)
  - llm_ms / tokens   — one Claude consult: wall time + input/output tokens
                        (skipped without ANTHROPIC_API_KEY or with --no-llm)

Usage:
    python benchmarks/benchmark.py --duration 120
    python benchmarks/benchmark.py --duration 60 --no-llm --output runs/baseline.json

Run from the project root (V:\\AI-ML\\REAgents). Works without the game open —
capture falls back to the primary monitor, which is fine for latency numbers
(the pixel cost is identical whether the frame shows RE4 or your desktop).
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running both as `python benchmarks/benchmark.py` and `-m benchmarks.benchmark`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1))))
    return ordered[k]


def _stats(samples_ms):
    """Summary stats for a list of millisecond samples."""
    if not samples_ms:
        return {"n": 0}
    return {
        "n": len(samples_ms),
        "mean": round(statistics.fmean(samples_ms), 2),
        "p50": round(_percentile(samples_ms, 50), 2),
        "p95": round(_percentile(samples_ms, 95), 2),
        "min": round(min(samples_ms), 2),
        "max": round(max(samples_ms), 2),
    }


def run_benchmark(duration_s: float, use_llm: bool, config_path: str) -> dict:
    from dotenv import load_dotenv
    load_dotenv()

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_s": duration_s,
        "config": config_path,
        "python": sys.version.split()[0],
    }

    # ── Environment info ──────────────────────────────────────────────────────
    import torch
    results["env"] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }

    from capture import ScreenCapture
    from perception import PerceptionSystem

    cap = ScreenCapture(config_path)
    eyes = PerceptionSystem(config_path)

    # Let the grabber thread warm up
    time.sleep(1.0)

    # ── 1. Capture throughput ─────────────────────────────────────────────────
    print(f"[1/4] Capture throughput ({duration_s:.0f}s)…")
    t_end = time.time() + duration_s
    frames = 0
    while time.time() < t_end:
        _ = cap.get_frame()
        frames += 1
    results["capture_fps"] = round(frames / duration_s, 1)

    # ── 2. Perception component latencies ─────────────────────────────────────
    n_iters = max(10, int(duration_s / 2))
    print(f"[2/4] Perception latencies ({n_iters} iterations)…")
    yolo_ms, ocr_ms, health_ms, death_ms, total_ms = [], [], [], [], []

    for i in range(n_iters):
        frame = cap.get_frame()

        t0 = time.perf_counter()
        dets = eyes.detect_objects(frame)
        t1 = time.perf_counter()
        hud = eyes.read_hud(frame)
        t2 = time.perf_counter()
        _ = eyes.get_health_percentage(frame)
        t3 = time.perf_counter()
        _ = eyes.detect_death_screen(frame)
        t4 = time.perf_counter()

        yolo_ms.append((t1 - t0) * 1000)
        ocr_ms.append((t2 - t1) * 1000)
        health_ms.append((t3 - t2) * 1000)
        death_ms.append((t4 - t3) * 1000)
        total_ms.append((t4 - t0) * 1000)
        if (i + 1) % max(1, n_iters // 5) == 0:
            print(f"    {i + 1}/{n_iters}  last tick: {total_ms[-1]:.0f} ms")

    results["yolo_ms"] = _stats(yolo_ms)
    results["ocr_ms"] = _stats(ocr_ms)
    results["health_ms"] = _stats(health_ms)
    results["death_screen_ms"] = _stats(death_ms)
    results["perception_tick_ms"] = _stats(total_ms)

    # ── 3. Effective decision rate ────────────────────────────────────────────
    # Simulates the env.step() perception cadence: fresh frame every step,
    # perception re-run at perception_fps. Reports achievable steps/sec.
    print("[3/4] Simulated step loop (30s)…")
    import yaml
    with open(config_path) as f:
        _cfg = yaml.safe_load(f)
    hold = float(_cfg["rl_hyperparameters"].get("action_hold_seconds", 0.04))
    perc_fps = float(_cfg["game_settings"].get("perception_fps", 10))
    perc_interval = 1.0 / perc_fps

    steps = 0
    last_perc = 0.0
    t_end = time.time() + 30
    while time.time() < t_end:
        t0 = time.time()
        _ = cap.get_frame()
        if t0 - last_perc >= perc_interval:
            _ = eyes.detect_objects(cap.get_frame())
            _ = eyes.read_hud(cap.get_frame())
            last_perc = t0
        time.sleep(hold)
        steps += 1
    results["sim_step_rate_hz"] = round(steps / 30, 1)
    results["action_hold_seconds"] = hold

    # ── 4. LLM consult (optional) ─────────────────────────────────────────────
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        print("[4/4] LLM consult timing…")
        try:
            from llm_agent import ClaudeAdvisor
            from shared_state import SharedState

            shared = SharedState()
            shared.update(frame=cap.get_frame(), hud=eyes.read_hud(cap.get_frame()),
                          detections=[])
            advisor = ClaudeAdvisor(config_path)

            consult_ms, usages = [], []
            for _ in range(2):   # two consults: cold + warm
                t0 = time.perf_counter()
                advisor.consult(shared, episode_step=0)
                consult_ms.append((time.perf_counter() - t0) * 1000)
                usage = getattr(advisor, "_last_usage", None)
                if usage:
                    usages.append(usage)

            results["llm_ms"] = _stats(consult_ms)
            results["llm_usage"] = usages
        except Exception as exc:
            results["llm_error"] = str(exc)
            print(f"    LLM consult failed: {exc}")
    else:
        print("[4/4] LLM consult skipped (--no-llm or no API key).")
        results["llm_ms"] = None

    cap.stop()
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="REAgents performance benchmark")
    p.add_argument("--duration", type=float, default=120, help="Capture benchmark seconds")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--no-llm", action="store_true", help="Skip the Claude consult benchmark")
    p.add_argument("--output", default=None, help="Output path (default: runs/benchmark_<ts>.json)")
    args = p.parse_args()

    results = run_benchmark(args.duration, not args.no_llm, args.config)

    out = Path(args.output) if args.output else (
        Path("runs") / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n══ Benchmark results ══")
    print(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
