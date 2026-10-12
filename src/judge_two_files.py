"""
Independent LLM-as-judge scoring - TWO-FILE version.   [FIXED VERSION]

Uses the semantic judge from judge_independent.py (prompt unchanged).

FIXES IN THIS VERSION
1. Judging failures / malformed judge JSON are never stored as 0.0.  The numeric
   score is recovered when possible (see judge_independent.parse_judge_output);
   otherwise the row is left OUT of the checkpoint (so a re-run retries it) and
   listed in the "Issues" sheet.
2. Image ids and labels are normalised before use: whitespace stripped, trailing
   "." removed from ids (e.g. "HEINRICH." -> "HEINRICH"), condition stripped
   ("normal " -> "normal"), run_id converted to int.
3. Ground-truth matching is exact first, then a UNIQUE id-suffix match.  An
   ambiguous suffix match is reported instead of silently taking the first hit.
4. Duplicate (image, condition, injection, run) keys and missing run ids in the
   responses file are reported in the "Issues" sheet (a duplicate is NOT scored
   twice and NOT silently dropped without a record).
5. Summary reports N scored versus expected (images x max run) per cell group,
   so the exact coverage can be quoted in Methods.
6. The ollama judge model must be given explicitly with --judge_model.
"""

import argparse
import os

import pandas as pd
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

from judge_independent import JUDGE_PROMPT, call_groq, call_ollama, parse_judge_output, judge_pair, resolve_judge_model


def norm_id(x) -> str:
    return str(x).strip().rstrip(".").strip()


def load_gt_map(gt_source_path: str) -> dict:
    """Read only Image_id -> GROUND TRUTH from the combined file (ignores
    its own response columns entirely)."""
    df = pd.read_excel(gt_source_path, sheet_name="Sheet1", header=None)
    cols = df.iloc[1].tolist()
    data = df.iloc[2:].reset_index(drop=True)
    data.columns = cols
    return {norm_id(k): str(v) for k, v in zip(data["Image_id"], data["GROUND TRUTH"])}


def match_gt(original_image_id: str, gt_map: dict):
    """Exact id match first; otherwise a UNIQUE suffix match
    (File 2's Image_id = prefix + File 1's Original_image_id).
    Returns (gt, matched_id, status) with status in {"ok", "none", "ambiguous"}."""
    if original_image_id in gt_map:
        return gt_map[original_image_id], original_image_id, "ok"
    hits = [k for k in gt_map if k.endswith(original_image_id)]
    if len(hits) == 1:
        return gt_map[hits[0]], hits[0], "ok"
    return None, None, ("ambiguous" if hits else "none")


def prepare_responses(responses: pd.DataFrame):
    """Normalise labels and report data problems.  Returns (clean_df, issues)."""
    issues = []
    r = responses.copy()
    r["Original_image_id"] = r["Original_image_id"].map(norm_id)
    r["condition"] = r["condition"].astype(str).str.strip()
    r["injection_type"] = r["injection_type"].astype(str).str.strip()
    r["run_num"] = pd.to_numeric(r["run_id"], errors="coerce")

    keycols = ["Original_image_id", "condition", "injection_type"]
    # missing run ids -> report; give them the next unused run number in their cell group
    for idx in r.index[r["run_num"].isna()]:
        grp = r[(r[keycols] == r.loc[idx, keycols]).all(axis=1)]
        nxt = int(grp["run_num"].max(skipna=True) if grp["run_num"].notna().any() else 0) + 1
        r.loc[idx, "run_num"] = nxt
        issues.append((r.loc[idx, "Original_image_id"], r.loc[idx, "condition"], r.loc[idx, "injection_type"],
                       "", "MISSING run_id in responses file - assigned run %d; please verify" % nxt))
    r["run_num"] = r["run_num"].astype(int)

    dup = r.duplicated(keycols + ["run_num"], keep="first")
    for idx in r.index[dup]:
        issues.append((r.loc[idx, "Original_image_id"], r.loc[idx, "condition"], r.loc[idx, "injection_type"],
                       r.loc[idx, "run_num"], "DUPLICATE key in responses file - second copy NOT scored; fix labels at source"))
    return r[~dup].reset_index(drop=True), issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="Long-format responses file, e.g. gemini.xlsx")
    ap.add_argument("--gt_source", required=True, help="Combined file to pull GROUND TRUTH from, e.g. GEMINI.xlsx")
    ap.add_argument("--backend", choices=["groq", "ollama"], required=True)
    ap.add_argument("--judge_model", default=None, help="Judge model name (required for ollama)")
    ap.add_argument("--judge_note", default="", help="Free text recorded in the output, e.g. the GGUF source/quantisation")
    ap.add_argument("--model_name", default=None,
                    help="Override the evaluated model's name (default: read from the 'model' column)")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--checkpoint", default=None,
                    help="CSV path for incremental progress (default: <model_name>_checkpoint.csv). "
                         "Re-running the same command resumes; rows that failed are retried.")
    ap.add_argument("--fresh", action="store_true", help="Delete any existing checkpoint and re-score everything.")
    args = ap.parse_args()

    judge_model = resolve_judge_model(args.backend, args.judge_model)

    raw = pd.read_excel(args.responses, sheet_name="Sheet1")
    responses, issues = prepare_responses(raw)
    gt_map = load_gt_map(args.gt_source)

    model_name = args.model_name or (str(responses["model"].iloc[0]) if len(responses) else "Model")
    checkpoint_path = args.checkpoint or f"{model_name}_checkpoint.csv"
    if args.fresh and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"--fresh set: removed existing checkpoint {checkpoint_path}")

    cols = ["Original_image_id", "Matched_Image_id", "GT", "model", "injection_type", "condition",
            "run_id", "score", "band", "capped", "reason"]
    done_keys = set()
    if os.path.exists(checkpoint_path):
        prev = pd.read_csv(checkpoint_path)
        # Old checkpoints (made before this fix) may contain failed rows stored as 0.0
        # with an UNPARSEABLE/ERROR reason.  Those are NOT treated as done: they are
        # dropped here so they get judged again.
        failed = prev["reason"].astype(str).str.startswith(("UNPARSEABLE", "ERROR after", "FAILED after"))
        good = prev[prev["score"].notna() & ~failed].copy()
        # normalise labels written by older versions (trailing '.', trailing spaces)
        good["Original_image_id"] = good["Original_image_id"].map(norm_id)
        good["condition"] = good["condition"].astype(str).str.strip()
        good["injection_type"] = good["injection_type"].astype(str).str.strip()
        good = good[good["run_id"].notna()]
        good["run_id"] = good["run_id"].astype(float).astype(int)
        good = good.drop_duplicates(["Original_image_id", "condition", "injection_type", "run_id"], keep="last")
        if len(good) != len(prev):
            print(f"Checkpoint cleanup: {len(prev) - len(good)} old failed/duplicate rows removed; they will be re-judged.")
        good.to_csv(checkpoint_path, index=False)
        for _, p in good.iterrows():
            done_keys.add((p["Original_image_id"], p["condition"], p["injection_type"], int(p["run_id"])))
        print(f"Resuming: {len(done_keys)} already-scored rows kept from {checkpoint_path}.")
    else:
        pd.DataFrame(columns=cols).to_csv(checkpoint_path, index=False)

    n = len(responses)
    for i, row in responses.iterrows():
        orig_id, cond, inj, run = row["Original_image_id"], row["condition"], row["injection_type"], int(row["run_num"])
        key = (orig_id, cond, inj, run)
        if key in done_keys:
            continue

        explanation = row["explanation"]
        if pd.isna(explanation) or str(explanation).strip() in ("", "nan", "None"):
            issues.append((orig_id, cond, inj, run, "EMPTY explanation in responses file - not judged"))
            continue

        gt, matched, status = match_gt(orig_id, gt_map)
        if gt is None:
            issues.append((orig_id, cond, inj, run,
                           "NO ground-truth match" if status == "none" else "AMBIGUOUS ground-truth suffix match"))
            continue

        print(f"[{i+1}/{n}] {orig_id} (GT: {matched}) cond={cond} inj={inj} run={run}")
        score, band, reason, capped = judge_pair(gt, str(explanation), args.backend, judge_model)
        if score is None:
            issues.append((orig_id, cond, inj, run, f"JUDGE FAILED (not stored as 0; re-run to retry): {reason[:200]}"))
            continue

        pd.DataFrame([{"Original_image_id": orig_id, "Matched_Image_id": matched, "GT": gt,
                       "model": row["model"], "injection_type": inj, "condition": cond, "run_id": run,
                       "score": score, "band": band, "capped": capped, "reason": reason}]
                     ).to_csv(checkpoint_path, mode="a", header=False, index=False)

    if issues:
        print(f"\nWARNING: {len(issues)} issue(s) - see 'Issues' sheet:")
        for it in issues[:15]:
            print("   ", it)
    all_results = pd.read_csv(checkpoint_path)
    all_results = all_results[all_results["score"].notna()].to_dict("records")
    print(f"\nTotal scored (including resumed rows): {len(all_results)}")
    write_output(model_name, judge_model, args.judge_note, args.backend, all_results, issues, args.out_dir)


def write_output(model_name, judge_model, judge_note, backend, results, issues, out_dir):
    df = pd.DataFrame(results)
    wb = openpyxl.Workbook()

    ws1 = wb.active
    ws1.title = "Summary"
    ws1["A1"] = f"{model_name} — Independent LLM-as-Judge ({backend}: {judge_model})"
    ws1["A1"].font = Font(bold=True, size=13)
    ws1.merge_cells("A1:G1")
    if judge_note:
        ws1["A2"] = judge_note
        ws1["A2"].font = Font(italic=True, size=9)

    headers = ["Condition", "Injection Type", "Avg Score", "N runs scored", "Expected runs", "Min / Max", "Coverage"]
    for j, h in enumerate(headers, start=1):
        c = ws1.cell(row=3, column=j, value=h)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="4472C4")

    row_i = 4
    if len(df):
        n_img = df["Original_image_id"].nunique()
        max_run = int(df["run_id"].astype(float).max())
        expected = n_img * max_run
        for (cond, inj), g in df.groupby(["condition", "injection_type"]):
            s = g["score"].astype(float)
            ws1.cell(row_i, 1, cond); ws1.cell(row_i, 2, inj)
            ws1.cell(row_i, 3, round(s.mean(), 4)); ws1.cell(row_i, 4, int(s.count()))
            ws1.cell(row_i, 5, expected); ws1.cell(row_i, 6, f"{s.min():.2f} / {s.max():.2f}")
            ws1.cell(row_i, 7, f"{100 * s.count() / expected:.1f}%")
            row_i += 1
        ws1.cell(row_i + 1, 1, f"Images: {n_img}; runs per image and condition: {max_run}; "
                               f"scored {len(df)} of {expected * df.groupby(['condition','injection_type']).ngroups} planned.")
    ws1.cell(row_i + 2, 1, f"Rows not scored (excluded from means, never counted as 0): {len(issues)} - see 'Issues' sheet.")
    ws1.cell(row_i + 2, 1).font = Font(italic=True, size=9, color="C00000")
    ws1.cell(row_i + 3, 1, "Rubric: semantic closeness of the complete response to the complete GT. "
                           "No keyword or regex matching is used for scoring.")
    ws1.cell(row_i + 3, 1).font = Font(italic=True, size=9, color="555555")
    for col, w in zip("ABCDEFG", [16, 16, 12, 14, 14, 16, 10]):
        ws1.column_dimensions[col].width = w

    ws2 = wb.create_sheet("Per-Run Scores")
    headers2 = ["Original_image_id", "GT", "Condition", "Injection Type", "Run", "Score", "Band", "Capped?", "Reason"]
    for j, h in enumerate(headers2, start=1):
        c = ws2.cell(row=1, column=j, value=h)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="4472C4")
    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for i, r in enumerate(results, start=2):
        vals = [r["Original_image_id"], r["GT"], r["condition"], r["injection_type"],
                int(float(r["run_id"])), float(r["score"]), r["band"], "", r["reason"]]
        for j, v in enumerate(vals, start=1):
            cell = ws2.cell(row=i, column=j, value=v)
            cell.border = border
            if j == 9:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                cell.font = Font(size=8)
    for col, w in zip("ABCDEFGHI", [30, 20, 11, 14, 6, 8, 11, 9, 50]):
        ws2.column_dimensions[col].width = w
    ws2.freeze_panes = "A2"

    ws3 = wb.create_sheet("Issues")
    for j, h in enumerate(["Original_image_id", "Condition", "Injection Type", "Run", "Problem"], start=1):
        c = ws3.cell(row=1, column=j, value=h)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="C00000")
    for i, row in enumerate(issues, start=2):
        for j, v in enumerate(row, start=1):
            ws3.cell(row=i, column=j, value=v)
    ws3.column_dimensions["A"].width = 30
    ws3.column_dimensions["E"].width = 90

    out_path = f"{out_dir}/{model_name}_LLM_judge_independent_scores.xlsx"
    wb.save(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
