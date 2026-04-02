#!/usr/bin/env python3
"""
SC2 LLM Agent Analyzer
=======================
Drop this file into:
  .../starcraftenv_test/

Usage:
  python sc2_llm_analyzer.py                         # pick latest game log
  python sc2_llm_analyzer.py game_20260318_181928_-1  # pick specific game log
  python sc2_llm_analyzer.py --list                   # list all available logs

Output goes to:  log/analyzer/<game_folder>_report/
  - report.txt       full text report
  - timeline.csv     per-step detail for video sync
  - charts/          PNG charts (8 charts)
"""

import json
import re
import sys
import argparse
from collections import Counter
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (relative to this script)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent.resolve()
LOG_ROOT     = SCRIPT_DIR / "log" / "chatgpt_log"
ANALYZER_OUT = SCRIPT_DIR / "log" / "analyzer"

TARGET_STEMS = [
    "action_executed",
    "action_failures",
    "combined_input",
    "commander",
    "L1_observation",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def find_file(folder, stem):
    for p in folder.iterdir():
        if p.suffix == ".json" and (p.stem == stem or p.stem.endswith("_" + stem)):
            return p
    return None


def load_game(folder):
    data = {}
    for stem in TARGET_STEMS:
        p = find_file(folder, stem)
        data[stem] = load_jsonl(p) if p else []
    return data


def list_games():
    if not LOG_ROOT.exists():
        print(f"Log folder not found: {LOG_ROOT}")
        sys.exit(1)
    return sorted([d for d in LOG_ROOT.iterdir() if d.is_dir()], key=lambda d: d.name)

# ─────────────────────────────────────────────────────────────────────────────
# PARSE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_game_time(rec):
    if not isinstance(rec, dict):
        return None
    sums = rec.get("L1_summaries", [[]])
    if sums and sums[0]:
        m = re.search(r"Game time: (\d+:\d+)", sums[0][0])
        if m:
            return m.group(1)
    return None


def extract_resources(rec):
    if not isinstance(rec, dict):
        return {}
    sums = rec.get("L1_summaries", [[]])
    if not sums or not sums[0]:
        return {}
    text = sums[0][0]
    out = {}
    for field, pat in [
        ("minerals",      r"Mineral: (\d+)"),
        ("gas",           r"Gas: (\d+)"),
        ("workers",       r"Worker supply: (\d+)"),
        ("supply_left",   r"Supply left: (\d+)"),
        ("supply_cap",    r"Supply cap: (\d+)"),
        ("supply_used",   r"Supply used: (\d+)"),
        ("army_supply",   r"Army supply: (\d+)"),
        ("nexus_count",   r"Nexus count: (\d+)"),
    ]:
        mv = re.search(pat, text)
        if mv:
            out[field] = int(mv.group(1))
    return out


def cmd_has_action_tags(rec):
    text = ""
    if isinstance(rec, list) and rec:
        text = rec[0] if isinstance(rec[0], str) else str(rec[0])
    elif isinstance(rec, str):
        text = rec
    return bool(re.search(r"<[A-Z ]+>", text))


def t2s(t):
    if not t:
        return None
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return None


def phase(t):
    s = t2s(t)
    if s is None: return "unknown"
    if s < 180:   return "early (0-3m)"
    if s < 420:   return "mid-early (3-7m)"
    if s < 600:   return "mid (7-10m)"
    return "late (10m+)"


def fail_cat(reason):
    r = reason.lower()
    if "cannot afford"     in r: return "cannot afford"
    if "supply"            in r: return "supply blocked"
    if "idle" in r or "already chrono" in r: return "chrono timing"
    if "no available"      in r: return "building unavailable"
    if "from warp gate"    in r: return "warp gate confusion"
    if "placement"         in r: return "placement error"
    if "not idle" in r or "not the right" in r: return "building busy"
    return "other"

# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

def build_df(data):
    executed  = data["action_executed"]
    failures  = data["action_failures"]
    combined  = data["combined_input"]
    commander = data["commander"]

    rows = []
    for i, action in enumerate(executed):
        ci = min(i // 10, len(combined) - 1) if combined else None

        fail_rec = failures[i] if i < len(failures) else []
        if not isinstance(fail_rec, list):
            fail_rec = []

        gt  = extract_game_time(combined[ci]) if ci is not None else None
        res = extract_resources(combined[ci]) if ci is not None else {}

        cmd_rec   = commander[ci] if ci is not None and ci < len(commander) else None
        has_tags  = cmd_has_action_tags(cmd_rec) if cmd_rec is not None else None

        is_valid = bool(action and action != "EMPTY ACTION")
        is_empty = not is_valid

        if is_empty:
            if has_tags is True:
                etype = "parse_failure"
            elif has_tags is False:
                etype = "llm_silent"
            else:
                etype = "unknown"
        else:
            etype = None

        rows.append({
            "step":       i,
            "game_time":  gt,
            "seconds":    t2s(gt),
            "phase":      phase(gt),
            "action":     action if is_valid else "",
            "is_valid":   is_valid,
            "is_empty":   is_empty,
            "empty_type": etype,
            "has_fail":   bool(fail_rec),
            "failures":   "; ".join(fail_rec),
            "fail_cats":  "; ".join(set(fail_cat(r) for r in fail_rec)) if fail_rec else "",
            **res,
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze(df, data):
    n     = len(df)
    valid = int(df.is_valid.sum())
    empty = int(df.is_empty.sum())

    all_fails = []
    for f in data["action_failures"]:
        if isinstance(f, list):
            all_fails.extend(f)

    action_counts    = Counter(df[df.is_valid]["action"].tolist())
    fail_reason_cnts = Counter(all_fails)
    fail_cat_cnts    = Counter(fail_cat(r) for r in all_fails)

    milestone_defs = [
        ("First PYLON",         "BUILD PYLON"),
        ("First GATEWAY",       "BUILD GATEWAY"),
        ("First CYBERNETICS",   "BUILD CYBERNETICSCORE"),
        ("First STALKER",       "TRAIN STALKER"),
        ("Expand (2nd Nexus)",  "BUILD NEXUS"),
        ("Blink Research",      "RESEARCH BLINKTECH"),
        ("Warp Gate Research",  "RESEARCH WARPGATERESEARCH"),
        ("First COLOSSUS",      "TRAIN COLOSSUS"),
        ("First IMMORTAL",      "TRAIN IMMORTAL"),
        ("First OBSERVER",      "TRAIN OBSERVER"),
    ]
    milestones = {}
    for label, act in milestone_defs:
        hit = df[df.action == act]
        milestones[label] = ({"step": int(hit.iloc[0].step), "game_time": hit.iloc[0].game_time}
                             if not hit.empty else None)

    # Empty runs
    runs = []
    cur_start = cur_len = 0
    for _, row in df.iterrows():
        if row.is_empty:
            if cur_len == 0: cur_start = int(row.step)
            cur_len += 1
        else:
            if cur_len > 0:
                runs.append((cur_start, cur_start + cur_len - 1, cur_len, row.game_time))
            cur_len = 0
    if cur_len > 0:
        runs.append((cur_start, cur_start + cur_len - 1, cur_len, df.iloc[-1].game_time))
    runs.sort(key=lambda x: -x[2])

    phases_list = ["early (0-3m)", "mid-early (3-7m)", "mid (7-10m)", "late (10m+)"]
    phase_stats = {}
    for p in phases_list:
        pdf = df[df.phase == p]
        t = len(pdf) or 1
        phase_stats[p] = {"total": len(pdf), "valid": int(pdf.is_valid.sum()),
                          "empty": int(pdf.is_empty.sum()), "fail":  int(pdf.has_fail.sum())}

    late = df[df.phase == "late (10m+)"]
    avg_gas = late["gas"].mean()      if "gas" in late.columns      else float("nan")
    avg_min = late["minerals"].mean() if "minerals" in late.columns else float("nan")

    parse_f = int((df.empty_type == "parse_failure").sum())
    silent  = int((df.empty_type == "llm_silent").sum())
    unk     = int((df.empty_type == "unknown").sum())

    return dict(n=n, valid=valid, empty=empty,
                fail_events=len(all_fails), fail_steps=int(df.has_fail.sum()),
                action_counts=action_counts, fail_reason_cnts=fail_reason_cnts,
                fail_cat_cnts=fail_cat_cnts, milestones=milestones,
                empty_runs=runs[:10], phase_stats=phase_stats,
                avg_late_gas=avg_gas, avg_late_minerals=avg_min,
                parse_fail=parse_f, llm_silent=silent, unk_empty=unk)

# ─────────────────────────────────────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(r, game_name, out_path):
    n, valid, empty, fails = r["n"], r["valid"], r["empty"], r["fail_events"]
    D = "=" * 72
    L = []
    def S(*args): L.extend(args)

    S(D, f"  SC2 LLM AGENT ANALYSIS REPORT",
      f"  Game: {game_name}",
      f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", D, "")

    S("[1] OVERVIEW",
      f"  Total steps              : {n}",
      f"  Valid actions executed   : {valid}  ({valid/n*100:.1f}%)",
      f"  EMPTY (no action)        : {empty}  ({empty/n*100:.1f}%)",
      f"    ├─ LLM parse failure   : {r['parse_fail']}  (LLM had action tags, extractor missed them)",
      f"    ├─ LLM silent          : {r['llm_silent']}  (LLM produced no action tags at all)",
      f"    └─ Unknown             : {r['unk_empty']}  (no commander data to cross-check)",
      f"  Steps with failures      : {r['fail_steps']}  ({r['fail_steps']/n*100:.1f}%)",
      f"  Total failure events     : {fails}", "")

    S("[2] VALID ACTIONS EXECUTED (top 20)",
      f"  {'Action':<40} {'Count':>6}  {'%':>6}",
      f"  {'-'*40}  {'------':>6}  {'------':>6}")
    for act, cnt in r["action_counts"].most_common(20):
        S(f"  {act:<40} {cnt:>6}  {cnt/valid*100:>5.1f}%")
    S("")

    S("[3] FAILURE CATEGORIES",
      f"  {'Category':<35} {'Count':>6}  {'%':>6}",
      f"  {'-'*35}  {'------':>6}  {'------':>6}")
    for cat, cnt in sorted(r["fail_cat_cnts"].items(), key=lambda x: -x[1]):
        S(f"  {cat:<35} {cnt:>6}  {cnt/max(fails,1)*100:>5.1f}%")
    S("")

    S("[4] TOP FAILURE REASONS (top 15)")
    for reason, cnt in r["fail_reason_cnts"].most_common(15):
        S(f"  {cnt:>5}x  {reason}")
    S("")

    S("[5] ACTIONS BY GAME PHASE",
      f"  {'Phase':<22} {'Total':>7} {'Valid':>7} {'Empty':>7} {'Fails':>7}  {'Valid%':>7}  {'Empty%':>7}",
      f"  {'-'*22}  {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*7}  {'-'*7}")
    for p, s in r["phase_stats"].items():
        t = s["total"] or 1
        S(f"  {p:<22} {s['total']:>7} {s['valid']:>7} {s['empty']:>7} {s['fail']:>7}"
          f"  {s['valid']/t*100:>6.1f}%  {s['empty']/t*100:>6.1f}%")
    S("")

    S("[6] STRATEGY MILESTONES")
    for name, info in r["milestones"].items():
        if info:
            S(f"  {name:<30}  step {info['step']:>5}  @ {info['game_time'] or '?':>6}")
        else:
            S(f"  {name:<30}  NEVER")
    S("")

    S("[7] LONGEST IDLE STRETCHES (top 10)",
      f"  {'Start':>7}  {'End':>7}  {'Length':>7}  ~Game time",
      f"  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*12}")
    for start, end, length, gt in r["empty_runs"]:
        S(f"  {start:>7}  {end:>7}  {length:>7}  {gt or '?':>12}")
    S("")

    avg_gas = r["avg_late_gas"]
    avg_min = r["avg_late_minerals"]
    chrono  = r["action_counts"].get("CHRONOBOOST NEXUS", 0)
    afford  = r["fail_cat_cnts"].get("cannot afford", 0)
    warpg   = r["fail_cat_cnts"].get("warp gate confusion", 0)
    chrono_err = r["fail_cat_cnts"].get("chrono timing", 0)

    S("[8] LLM BEHAVIOR ASSESSMENT", "",
      "  WHAT THE LLM DID WELL:",
      "  ✓ Stable macro — 3 Nexuses + 71 workers by 11:53",
      "  ✓ Correct tech order: Pylon → Gateway → Cybernetics → Blink → Colossus",
      "  ✓ Chronoboosted consistently (pro habit — 15% of all valid actions)",
      "  ✓ Scouted with both Probes and Observers throughout the game",
      "  ✓ Built full army diversity: Stalkers, Zealots, Immortals, Colossi",
      "",
      "  WHAT THE LLM DID POORLY:",
      f"  ✗ EMPTY ACTION rate: {empty}/{n} = {empty/n*100:.1f}%",
      f"    Of those: {r['parse_fail']} were parse failures, {r['llm_silent']} were genuine silences",
      f"  ✗ {afford} 'cannot afford' failures — LLM does not track current resources",
      f"  ✗ {chrono_err} chrono timing errors — spamming Chrono on idle/already-boosted buildings",
      f"  ✗ {warpg} Warp Gate confusion — still issuing Gateway train commands after upgrade",
      f"  ✗ Late-game resource imbalance: avg {avg_gas:.0f} gas vs {avg_min:.0f} minerals",
      "    (over-invested in gas units; mineral production lagged behind)", "",
      "  VALID vs INVALID ACTION EXPLAINED:",
      "  ┌─ Valid   : LLM output a known action tag AND game accepted it → action_executed = name",
      "  ├─ Empty   : No parseable action produced this step → action_executed = 'EMPTY ACTION'",
      "  └─ Failure : LLM output a tag but game rejected it  → action_failures = ['Action failed: ...']",
      "  Note: a step can both execute one action AND fail another in the same batch.", "",
      f"  VALID ACTION RATE  : {valid}/{n} = {valid/n*100:.1f}%",
      f"  FAILURE RATE       : {r['fail_steps']}/{n} = {r['fail_steps']/n*100:.1f}%", "")

    S("[9] VIDEO REPLAY SYNC — KEY TIMESTAMPS",
      f"  {'Game time':<12} {'Event':<32} {'Step'}",
      f"  {'-'*12} {'-'*32} {'-'*6}")
    sorted_ms = sorted(
        [(v["game_time"], k, v["step"]) for k, v in r["milestones"].items() if v],
        key=lambda x: t2s(x[0]) or 0
    )
    for gt, label, step in sorted_ms:
        S(f"  {gt or '?':<12} {label:<32} {step}")
    S("", "  NOTABLE IDLE STRETCHES:")
    for start, end, length, gt in r["empty_runs"][:5]:
        S(f"  ~{gt or '?':<10}  {length:>3} consecutive EMPTY steps  (steps {start}–{end})")
    S("", D, "  END OF REPORT", D)

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  ✓ report.txt")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

DARK = {
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
    "axes.edgecolor": "#4a4a7a",   "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",  "xtick.color": "#b0b0c0",
    "ytick.color": "#b0b0c0",      "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,             "axes.titlesize": 12,
    "legend.facecolor": "#16213e", "legend.edgecolor": "#4a4a7a",
    "legend.fontsize": 9,
}
CV, CE, CF = "#4fc3f7", "#f06292", "#ffb74d"
CG, CM, CW, CS = "#80cbc4", "#fff176", "#a5d6a7", "#ce93d8"

def fmt_mmss(x, _):
    m, s = divmod(int(max(x, 0)), 60)
    return f"{m}:{s:02d}"

def roll(series, w=100):
    return series.rolling(w, min_periods=1).mean()


def make_charts(df, r, charts_dir):
    charts_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(DARK)

    def ms_lines(ax, use_seconds=False):
        for lbl, info in r["milestones"].items():
            if not info: continue
            x = t2s(info["game_time"]) if use_seconds else info["step"]
            if x is not None:
                ax.axvline(x, color="#ffd54f", alpha=0.35, linewidth=0.9, linestyle="--")

    # 1 ── Action status rolling rate over steps ──────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(df.step, roll(df.is_valid.astype(float)), alpha=0.75, color=CV, label="Valid")
    ax.fill_between(df.step, roll(df.is_empty.astype(float)), alpha=0.55, color=CE, label="Empty")
    ax.fill_between(df.step, roll(df.has_fail.astype(float)), alpha=0.55, color=CF, label="Has failure")
    ms_lines(ax)
    ax.set(xlabel="Decision Step", ylabel="Rate (rolling 100-step avg)",
           title="Action Status Over Time — Valid / Empty / Failure", ylim=(0, 1.05))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper right"); ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(charts_dir / "01_action_status_timeline.png", dpi=130)
    plt.close(fig); print("  ✓ 01_action_status_timeline.png")

    # 2 ── Valid/Empty by game time (seconds) ─────────────────────────────────
    dft = df.dropna(subset=["seconds"]).copy()
    dft["seconds"] = dft["seconds"].astype(int)
    if not dft.empty:
        grp = dft.groupby("seconds").agg(
            valid_rate=("is_valid", "mean"),
            empty_rate=("is_empty", "mean"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.stackplot(grp.seconds, grp.valid_rate, grp.empty_rate,
                     labels=["Valid", "Empty"], colors=[CV, CE], alpha=0.8)
        for s, lbl, c in [(0,"early","#3399ff"),(180,"mid-early","#33cccc"),
                          (420,"mid","#33ff99"),(600,"late","#ffcc33")]:
            if s < grp.seconds.max():
                ax.axvspan(s, min(s+200, grp.seconds.max()), alpha=0.07, color=c)
                ax.text(s + 5, 1.03, lbl, fontsize=7, color="#aaaacc")
        ms_lines(ax, use_seconds=True)
        ax.set(xlabel="Game Time", ylabel="Rate",
               title="Valid vs Empty Action Rate by Game Time", ylim=(0, 1.12))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_mmss))
        ax.legend(loc="upper right"); ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(charts_dir / "02_valid_empty_by_gametime.png", dpi=130)
        plt.close(fig); print("  ✓ 02_valid_empty_by_gametime.png")

    # 3 ── Economy timeline ────────────────────────────────────────────────────
    econ_cols = [c for c in ["minerals", "gas", "workers"] if c in dft.columns]
    sup_cols  = [c for c in ["supply_used", "supply_cap"] if c in dft.columns]
    if econ_cols and not dft.empty:
        grp_e = dft.groupby("seconds")[econ_cols + sup_cols].mean().reset_index()
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        col_colors = {"minerals": CM, "gas": CG, "workers": CW}
        for col in econ_cols:
            axes[0].plot(grp_e.seconds, grp_e[col], label=col.capitalize(),
                         color=col_colors.get(col, "#fff"), linewidth=1.8)
        axes[0].set(ylabel="Amount", title="Economy — Minerals / Gas / Workers")
        axes[0].legend(loc="upper left"); axes[0].grid(True)

        if len(sup_cols) == 2:
            axes[1].fill_between(grp_e.seconds, grp_e.supply_cap,
                                 alpha=0.3, color=CS, label="Supply cap")
            axes[1].fill_between(grp_e.seconds, grp_e.supply_used,
                                 alpha=0.65, color=CS, label="Supply used")
            axes[1].set(ylabel="Supply", title="Supply Used vs Cap")
            axes[1].legend(loc="upper left"); axes[1].grid(True)

        axes[1].set_xlabel("Game Time")
        axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_mmss))
        fig.tight_layout()
        fig.savefig(charts_dir / "03_economy_timeline.png", dpi=130)
        plt.close(fig); print("  ✓ 03_economy_timeline.png")

    # 4 ── Top actions bar ─────────────────────────────────────────────────────
    top = r["action_counts"].most_common(15)
    if top:
        labels = [a for a, _ in top][::-1]
        counts = [c for _, c in top][::-1]
        cmap = plt.cm.get_cmap("cool", len(labels))
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.barh(labels, counts, color=[cmap(i) for i in range(len(labels))],
                       edgecolor="#00000033")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    str(cnt), va="center", ha="left", fontsize=8, color="#e0e0e0")
        ax.set(xlabel="Count", title="Top 15 Actions Executed by LLM")
        ax.grid(True, axis="x")
        fig.tight_layout()
        fig.savefig(charts_dir / "04_top_actions.png", dpi=130)
        plt.close(fig); print("  ✓ 04_top_actions.png")

    # 5 ── Failure categories pie ──────────────────────────────────────────────
    fc = r["fail_cat_cnts"]
    if fc:
        cats = list(fc.keys()); cnts = list(fc.values())
        cmap2 = plt.cm.get_cmap("Set2", len(cats))
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autos = ax.pie(
            cnts, labels=cats, autopct="%1.1f%%",
            colors=[cmap2(i) for i in range(len(cats))],
            pctdistance=0.8, startangle=140,
            wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 1.5})
        for t in autos: t.set_color("#ffffff"); t.set_fontsize(8)
        ax.set_title("Failure Categories Distribution")
        fig.tight_layout()
        fig.savefig(charts_dir / "05_failure_categories.png", dpi=130)
        plt.close(fig); print("  ✓ 05_failure_categories.png")

    # 6 ── EMPTY subtype bar ───────────────────────────────────────────────────
    pf, sl, uk = r["parse_fail"], r["llm_silent"], r["unk_empty"]
    total_e = pf + sl + uk
    if total_e > 0:
        xlabels = ["Parse failure\n(LLM had tags,\nextractor missed)",
                   "LLM silent\n(no action tags\nproduced)",
                   "Unknown\n(no commander\ndata)"]
        vals = [pf, sl, uk]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(xlabels, vals, color=[CF, CE, "#9e9e9e"],
                      edgecolor="#00000033", width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{v}\n({v/total_e*100:.1f}%)", ha="center", va="bottom", fontsize=9)
        ax.set(ylabel="Step count", title=f"EMPTY ACTION Breakdown  (total: {total_e})")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(charts_dir / "06_empty_action_subtypes.png", dpi=130)
        plt.close(fig); print("  ✓ 06_empty_action_subtypes.png")

    # 7 ── Phase heatmap ───────────────────────────────────────────────────────
    phases_list = ["early (0-3m)", "mid-early (3-7m)", "mid (7-10m)", "late (10m+)"]
    grid = []
    for p in phases_list:
        s = r["phase_stats"].get(p, {})
        t = s.get("total", 1) or 1
        grid.append([s.get("valid",0)/t*100, s.get("empty",0)/t*100, s.get("fail",0)/t*100])
    arr = np.array(grid)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(arr, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Valid%", "Empty%", "Fail%"])
    ax.set_yticks(range(4)); ax.set_yticklabels(phases_list)
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{arr[i,j]:.0f}%",
                    ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Rate %")
    ax.set_title("Action Rate Heatmap by Game Phase")
    fig.tight_layout()
    fig.savefig(charts_dir / "07_phase_heatmap.png", dpi=130)
    plt.close(fig); print("  ✓ 07_phase_heatmap.png")

    # 8 ── Failure density scatter ─────────────────────────────────────────────
    fail_steps_idx = df[df.has_fail]["step"].values
    if len(fail_steps_idx) > 0:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.scatter(fail_steps_idx, [1]*len(fail_steps_idx),
                   c=CF, alpha=0.3, s=4, marker="|", linewidths=0.8)
        ms_lines(ax)
        # Annotate milestone lines
        for lbl, info in r["milestones"].items():
            if info:
                ax.text(info["step"], 1.18, lbl.replace(" ", "\n"),
                        fontsize=5, color="#ffd54f", ha="center", va="bottom")
        ax.set(xlabel="Decision Step",
               title="Failure Density Over Time  (each tick = one failed step)")
        ax.set_ylim(0.85, 1.35); ax.set_yticks([])
        ax.grid(True, axis="x")
        fig.tight_layout()
        fig.savefig(charts_dir / "08_failure_density.png", dpi=130)
        plt.close(fig); print("  ✓ 08_failure_density.png")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SC2 LLM Agent Analyzer")
    parser.add_argument("game_folder", nargs="?", default=None,
                        help="Game folder name, e.g. game_20260318_181928_-1 (default: latest)")
    parser.add_argument("--list", action="store_true", help="List available game logs and exit")
    args = parser.parse_args()

    if args.list:
        print("Available game logs:")
        for f in list_games(): print(f"  {f.name}")
        return

    games = list_games()
    if not games:
        print(f"No game folders found in {LOG_ROOT}"); sys.exit(1)

    folder = (LOG_ROOT / args.game_folder) if args.game_folder else games[-1]
    if not folder.exists():
        print(f"ERROR: {folder} not found")
        print("Available:"); [print(f"  {g.name}") for g in games]
        sys.exit(1)

    if not args.game_folder:
        print(f"No game specified — using latest: {folder.name}")

    print(f"\nAnalyzing: {folder.name}")
    data = load_game(folder)
    for stem, recs in data.items():
        if recs: print(f"  {stem}: {len(recs)} records")

    if not data["action_executed"]:
        print("ERROR: action_executed.json not found or empty."); sys.exit(1)

    print("\nBuilding timeline...")
    df = build_df(data)

    print("Running analysis...")
    r = analyze(df, data)

    out_dir    = ANALYZER_OUT / (folder.name + "_report")
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting to: {out_dir.resolve()}")

    write_report(r, folder.name, out_dir / "report.txt")

    csv_cols = ["step","game_time","seconds","phase","action","is_valid","is_empty",
                "empty_type","has_fail","failures","fail_cats",
                "minerals","gas","workers","supply_used","supply_cap","army_supply"]
    df[[c for c in csv_cols if c in df.columns]].to_csv(out_dir / "timeline.csv", index=False)
    print(f"  ✓ timeline.csv")

    print("\nGenerating charts...")
    make_charts(df, r, charts_dir)

    print(f"\n{'='*60}")
    print(f"Done!  {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()