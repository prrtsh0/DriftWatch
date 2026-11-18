"""Dashboard for DriftWatch: renders monitor status and history as a self-contained HTML page.

Charts are drawn with matplotlib (already a dependency) and embedded as base64 PNGs,
so the dashboard needs no internet or CDN. The page is produced server-side by
`build_dashboard()` and served by FastAPI at `/dashboard`.
"""
import base64
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("driftwatch.dashboard")

ART = Path(os.getenv("DRIFTWATCH_ARTIFACTS", "artifacts"))
HISTORY = ART / "monitor_history.jsonl"

WARN, ALERT = 0.1, 0.25


def _fig_to_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_history(limit: int = 200) -> list[dict]:
    """Read past drift checks from the JSONL history file (newest last)."""
    if not HISTORY.exists():
        return []
    rows = []
    try:
        with open(HISTORY, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("dropping malformed history line")
    except OSError as e:  # pragma: no cover
        logger.warning("could not read history: %s", e)
    return rows[-limit:]


def append_history(record: dict) -> None:
    """Append one drift-check record to the history file (creates it on first use)."""
    record = dict(record)
    record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    try:
        ART.mkdir(exist_ok=True)
        with open(HISTORY, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as e:  # pragma: no cover
        logger.warning("could not append history: %s", e)


def _status_chart(history: list[dict]) -> str:
    """Bar of per-batch max_psi with warn/alert lines."""
    if not history:
        return ""
    ts = [r.get("timestamp", "")[11:19] for r in history]  # HH:MM:SS
    max_psi = [float(r.get("max_psi", 0.0)) for r in history]
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#d9534f" if v >= ALERT else "#f0ad4e" if v >= WARN else "#5cb85c" for v in max_psi]
    ax.bar(range(len(ts)), max_psi, color=colors)
    ax.axhline(ALERT, color="#d9534f", ls="--", lw=1.2, label=f"alert ({ALERT})")
    ax.axhline(WARN, color="#f0ad4e", ls=":", lw=1.2, label=f"warn ({WARN})")
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels(ts, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("max_psi")
    ax.set_title("Drift checks over time (max PSI per batch)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _feature_psi_chart(phi: dict) -> str:
    """Horizontal bar of per-feature PSI (top 10) with thresholds."""
    if not phi:
        return ""
    items = sorted(phi.items(), key=lambda kv: kv[1], reverse=True)[:10]
    feats = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d9534f" if v >= ALERT else "#f0ad4e" if v >= WARN else "#5cb85c" for v in vals]
    ax.barh(feats, vals, color=colors)
    ax.axvline(ALERT, color="#d9534f", ls="--", lw=1.2, label=f"alert ({ALERT})")
    ax.axvline(WARN, color="#f0ad4e", ls=":", lw=1.2, label=f"warn ({WARN})")
    ax.set_xlabel("PSI")
    ax.set_title("Per-feature PSI (latest batch vs reference)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def build_dashboard(latest: dict | None, history: list[dict]) -> str:
    status_chart = _status_chart(history)
    feature_psi = {}
    if latest and latest.get("feature_psi"):
        feature_psi = latest["feature_psi"]
    elif latest and latest.get("top_drifting_features"):
        feature_psi = {d["feature"]: d["psi"] for d in latest["top_drifting_features"]}
    feature_chart = _feature_psi_chart(feature_psi)

    cards = ""
    if latest:
        status = latest.get("status", "ok")
        badge = {"ok": "success", "warn": "warning", "alert": "danger"}.get(status, "secondary")
        cards = f"""
        <div class="card {badge}"><div class="k">status</div><div class="v">{status}</div></div>
        <div class="card"><div class="k">n_rows</div><div class="v">{latest.get('n_rows','-')}</div></div>
        <div class="card"><div class="k">max_psi</div><div class="v">{latest.get('max_psi','-')}</div></div>
        <div class="card"><div class="k">mean_psi</div><div class="v">{latest.get('mean_psi','-')}</div></div>
        <div class="card"><div class="k">mean_ks</div><div class="v">{latest.get('mean_ks','-')}</div></div>
        <div class="card"><div class="k">domain_auc</div><div class="v">{latest.get('domain_auc','-')}</div></div>
        <div class="card"><div class="k">pred_shift</div><div class="v">{latest.get('pred_shift','-')}</div></div>
        """

    latest_table = ""
    if latest and latest.get("top_drifting_features"):
        rows = "".join(
            f"<tr><td>{d['feature']}</td><td>{d['psi']:.4f}</td>"
            f"<td>{'alert' if d['psi'] >= ALERT else 'warn' if d['psi'] >= WARN else 'ok'}</td></tr>"
            for d in latest["top_drifting_features"]
        )
        latest_table = f"""
        <h3>Top drifting features (latest check)</h3>
        <table><tr><th>feature</th><th>PSI</th><th>status</th></tr>{rows}</table>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DriftWatch Dashboard</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #f6f8fa; color: #1b1f23; }}
  header {{ background: #24292f; color: #fff; padding: 14px 24px; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  header .sub {{ font-size: 12px; opacity: .75; }}
  .wrap {{ max-width: 960px; margin: 24px auto; padding: 0 16px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .card.success {{ border-left: 4px solid #2da44e; }} .card.warning {{ border-left: 4px solid #bf8700; }} .card.danger {{ border-left: 4px solid #cf222e; }}
  .k {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #57606a; }}
  .v {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  section {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; }}
  h3 {{ margin: 4px 0 10px; font-size: 15px; }}
  img {{ max-width: 100%; height: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #eaeef2; }}
  th {{ color: #57606a; font-weight: 600; }}
  .none {{ color: #8b949e; font-style: italic; }}
  footer {{ text-align: center; color: #8b949e; font-size: 12px; padding: 20px; }}
</style>
</head>
<body>
<header>
  <h1>DriftWatch</h1>
  <div class="sub">Label-free drift monitoring for the deployed model</div>
</header>
<div class="wrap">
  <div class="cards">{cards or '<div class="card"><div class="k">no data</div><div class="v">—</div></div>'}</div>
  <section><h3>Status history (max PSI per batch)</h3>{status_chart and f'<img src="data:image/png;base64,{status_chart}">' or '<div class="none">No drift checks recorded yet. POST to /drift or run the demo.</div>'}</section>
  <section>{feature_chart and f'<img src="data:image/png;base64,{feature_chart}">' or '<div class="none">No data yet.</div>'}</section>
  <section>{latest_table or '<div class="none">No per-feature data yet.</div>'}</section>
</div>
<footer>PSI thresholds — warn ≥ 0.10, alert ≥ 0.25. Built from monitor history (artifacts/monitor_history.jsonl).</footer>
</body>
</html>
"""
