#-*- coding: utf-8 -*-
"""Sanity checker for experiment logs: flags non-monotonic robustness
curves, collapsed runs, and unusually large cross-seed variance before
results are aggregated.

Usage: python check_anomaly.py run1.log run2.log   (or pipe logs via stdin)
Scans lines containing '>>>' produced by the experiment scripts.
"""
import sys, re
from collections import defaultdict

RED = "\033[91m"; YEL = "\033[93m"; GRN = "\033[92m"; RST = "\033[0m"

def load_lines(argv):
    txt = ""
    files = [a for a in argv if a not in ("-",)]
    if files:
        for f in files:
            try: txt += open(f, encoding="utf-8", errors="ignore").read() + "\n"
            except OSError as e: print(f"[skip] {f}: {e}")
    if not files or "-" in argv:
        if not sys.stdin.isatty(): txt += sys.stdin.read()
    return [l for l in txt.splitlines() if ">>>" in l]

def parse(line):
    """Extract the bracketed tag and all key=value numbers, including
    curve points such as f25=.. / vf50=.."""
    tag = re.search(r"\[([^\]]+)\]", line)
    tag = tag.group(1) if tag else line[:40]
    kv = {}
    for m in re.finditer(r"([A-Za-z_]?f\d+|vf\d+|r\d+|full|dropV|dropA|dropVideo|dropAudio|acc|macro)\s*=\s*([0-9]*\.?[0-9]+)", line):
        kv[m.group(1)] = float(m.group(2))
    return tag, kv, line

def curve_points(kv, prefix):
    pts = []
    for k, v in kv.items():
        m = re.fullmatch(rf"{prefix}(\d+)", k)
        if m: pts.append((int(m.group(1)), v))
    return sorted(pts)

def check_one(tag, kv, raw):
    issues = []
    # C1/C2: curve monotonicity + endpoint rebound
    for pref in ("f", "vf", "r"):
        pts = curve_points(kv, pref)
        if len(pts) >= 3:
            for i in range(1, len(pts)):
                if pts[i][1] > pts[i-1][1] + 0.03:
                    issues.append((f"C1 non-monotonic", f"{pref}{pts[i-1][0]}={pts[i-1][1]:.3f} -> {pref}{pts[i][0]}={pts[i][1]:.3f} rebound"))
            if pts[-1][1] > pts[-2][1] + 0.15:
                issues.append((f"C2 endpoint rebound", f"{pref}{pts[-2][0]}={pts[-2][1]:.3f} -> {pref}{pts[-1][0]}={pts[-1][1]:.3f} (possible mix-up: 100% frame-drop ~ modality-level)"))
    # C3: collapse
    for k, v in kv.items():
        if k in ("acc", "macro"): continue
        if v < 0.35:
            issues.append(("C3 collapse", f"{k}={v:.3f} < 0.35 (fails under this condition)"))
    # C5: metric mismatch
    f1_main = kv.get("full") or kv.get("dropV")
    if "acc" in kv and f1_main and f1_main > 0.60 and kv["acc"] < 0.55:
        issues.append(("C5 metric mismatch", f"F1={f1_main:.3f} high but acc={kv['acc']:.3f} low -> positive-skew / metric-inflation warning"))
    return issues

def main():
    lines = load_lines(sys.argv[1:])
    if not lines:
        print("(no '>>>' result lines to check)"); return
    # group by method to compute cross-seed std (C4)
    by_method = defaultdict(list)
    parsed = [parse(l) for l in lines]
    for tag, kv, raw in parsed:
        method = re.sub(r"[|]?seed\d+", "", tag).strip("| ")
        by_method[method].append(kv)

    any_issue = False
    print("="*70); print("Experiment result anomaly report"); print("="*70)
    for tag, kv, raw in parsed:
        iss = check_one(tag, kv, raw)
        if iss:
            any_issue = True
            print(f"\n{YEL}* [{tag}]{RST}")
            for code, msg in iss:
                print(f"   {RED}! {code}{RST}: {msg}")
    # C4: cross-seed outlier
    import statistics as st
    for method, rows in by_method.items():
        keys = set().union(*[r.keys() for r in rows]) if rows else set()
        for k in keys:
            vals = [r[k] for r in rows if k in r]
            if len(vals) >= 3:
                m, sd = st.mean(vals), st.pstdev(vals)
                if sd > 0.05:
                    for v in vals:
                        if abs(v - m) > 2.5 * sd:
                            any_issue = True
                            print(f"\n{YEL}* [{method}]{RST}\n   {RED}! C4 seed outlier{RST}: {k}={v:.3f} (mean {m:.3f}+/-{sd:.3f}, deviation >2.5 sigma)")
    print("\n" + "="*70)
    print((f"{GRN}PASS: no anomalies found{RST}") if not any_issue else f"{RED}Anomalies found above; verify manually before aggregating{RST}")

if __name__ == "__main__":
    main()
