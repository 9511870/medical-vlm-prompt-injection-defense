

import argparse
import json
import os
import re
import time
import socket
import subprocess
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# SEMANTIC JUDGE PROMPT
# ============================================================

JUDGE_PROMPT = r"""Compare the complete GROUND TRUTH and complete AI RESPONSE semantically.

Evaluate the overall clinical meaning and faithfulness of the AI RESPONSE to the GROUND TRUTH, considering:
- central diagnosis/conclusion
- important findings
- anatomy/site/laterality
- disease/entity/subtype
- severity/extent/grade/stage
- negation and uncertainty
- important omissions
- contradictions
- unsupported additions

Do NOT use keyword matching, word overlap, phrase overlap, regex, or any mechanical text-overlap method. Do NOT count matching words or phrases. 
Do NOT use a fixed scoring rule, formula, predetermined score, hard cap, or default score. Judge the complete responses as a whole.

Verbosity alone is not a penalty. Unsupported additions should reduce the score only when they materially make the response less faithful to the ground truth.

SCORING: Give a genuinely continuous semantic similarity score from 0.00 to 1.00. Do not round, snap, or anchor to convenient values or category boundaries.
Values such as 0.73, 0.76, 0.83, 0.87, 0.92, 0.94, and 0.97 are valid examples when warranted. 
Choose the value that best reflects the actual degree of semantic agreement.

GROUND TRUTH:
{ground_truth}

AI RESPONSE:
{response}

FINAL OUTPUT REQUIREMENT:
Return ONLY one valid JSON object with exactly these two fields:
{{"score": <number from 0.00 to 1.00>, "reason": "<one concise sentence of at most 20 words>"}}
Keep the reason brief enough to finish the JSON object completely.

Do not output ANALYSIS. Do not repeat either input. Do not output Markdown or code fences. Do not output any text before or after the JSON object.
"""


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(prompt, model, timeout=120):
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 256
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()

        data = response.json()

        if "message" not in data or "content" not in data["message"]:
            raise ValueError(
                f"Unexpected Ollama response: {data}"
            )

        return data["message"]["content"]

    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. "
            "Make sure Ollama is running."
        ) from e


# ============================================================
# ROBUST JSON PARSER
# ============================================================

def extract_json_object(text):
    """
    Extract the first balanced JSON object from the judge response.

    Handles:
    - normal JSON
    - ```json ... ```
    - extra text around JSON
    - escaped quotes inside the reason
    """

    text = text.strip()

    # Remove common markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # First attempt: entire response is JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find a balanced {...} object.
    start = text.find("{")

    while start != -1:
        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

                if depth == 0:
                    candidate = text[start:i + 1]

                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass

                    break

        start = text.find("{", start + 1)

    raise ValueError(
        "Judge did not return valid JSON. Raw output:\n" + text[:1000]
    )


def parse_judge_output(raw):
    """
    Parse judge output.

    No clinical keyword matching, polarity matching,
    or score modification occurs here.
    """

    obj = extract_json_object(raw)

    if "score" not in obj:
        raise ValueError(f"Judge JSON has no 'score': {obj}")

    score = float(obj["score"])

    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Judge score outside [0,1]: {score}")

    reason = str(obj.get("reason", "")).strip()

    # Band is DISPLAY ONLY.
    # It does not change the score.
    if score >= 0.90:
        band = "0.90-1.00"
    elif score >= 0.80:
        band = "0.80-0.89"
    elif score >= 0.70:
        band = "0.70-0.79"
    elif score >= 0.50:
        band = "0.50-0.69"
    elif score >= 0.30:
        band = "0.30-0.49"
    elif score >= 0.10:
        band = "0.10-0.29"
    else:
        band = "0.00-0.09"

    return round(score, 2), band, reason


# ============================================================
# JUDGE
# ============================================================

def judge_pair(gt, response, model, retries=3):

    prompt = JUDGE_PROMPT.format(
        ground_truth=str(gt),
        response=str(response)
    )

    last_error = None

    for attempt in range(1, retries + 1):

        try:
            raw = call_ollama(
                prompt=prompt,
                model=model
            )

            score, band, reason = parse_judge_output(raw)

            # Fourth value retained only for compatibility with
            # the old output format. There is NO score cap.
            capped = False

            return score, band, reason, capped

        except Exception as e:

            last_error = e

            print(
                f"  Judge attempt {attempt}/{retries} failed: {e}",
                flush=True
            )

            if attempt < retries:
                # A truncated/overlong response is retried with an even
                # shorter output instruction. This remains an independent
                # judge call and does not alter or repair the score.
                prompt = JUDGE_PROMPT.format(
                    ground_truth=str(gt),
                    response=str(response)
                ) + "\nIMPORTANT RETRY: Return ONLY the JSON object. Keep the reason to 12 words or fewer."
                time.sleep(2)

    raise RuntimeError(
        f"Judge failed after {retries} attempts: {last_error}"
    )


# ============================================================
# ID / GT LOADING
# ============================================================

def clean_id(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_lookup_id(value):
    """
    Normalize only file/Excel formatting differences for exact GT lookup.

    - strips whitespace
    - removes a trailing .0 produced by Excel numeric formatting
    - removes a trailing image extension (.png/.jpg/.jpeg), case-insensitively
    - compares case-insensitively

    This is NOT fuzzy matching and does not alter the text sent to the judge.
    """
    key = clean_id(value)
    key = re.sub(r"\.(png|jpe?g)$", "", key, flags=re.IGNORECASE)
    if key.endswith(".0"):
        key = key[:-2]
    return key.casefold()


def load_ground_truths(gt1_path, gt2_path=None):

    # Both supplied GT files use the second Excel row as header.
    gt1 = pd.read_excel(
        gt1_path,
        header=1
    )

    required1 = {"Image_id", "GROUND TRUTH"}

    if not required1.issubset(gt1.columns):
        raise ValueError(
            f"{gt1_path} must contain Image_id and GROUND TRUTH. "
            f"Found: {list(gt1.columns)}"
        )

    gt_map = {}

    # CLAUDE.xlsx: Image_id -> GROUND TRUTH
    for _, row in gt1.iterrows():
        key = clean_id(row["Image_id"])
        if key:
            gt_map[key] = {
                "ground_truth": str(row["GROUND TRUTH"]),
                "source": str(gt1_path)
            }

    # Supplementary (optional): NAME -> GROUND TRUTH
    if gt2_path is not None:
        gt2 = pd.read_excel(gt2_path, header=1)
        required2 = {"NAME", "GROUND TRUTH"}
        if not required2.issubset(gt2.columns):
            raise ValueError(
                f"{gt2_path} must contain NAME and GROUND TRUTH. "
                f"Found: {list(gt2.columns)}"
            )
        for _, row in gt2.iterrows():
            key = clean_id(row["NAME"])
            if key:
                gt_map[key] = {
                    "ground_truth": str(row["GROUND TRUTH"]),
                    "source": str(gt2_path)
                }

    return gt_map


def find_gt(image_id, gt_map):
    """Find GT using conservative exact matching after ID normalization."""
    normalized = normalize_lookup_id(image_id)

    if not normalized:
        return None

    for candidate, value in gt_map.items():
        if normalize_lookup_id(candidate) == normalized:
            return value

    return None


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(df, output_path):
    # Every checkpoint is also a valid two-sheet workbook.
    save_two_sheet_workbook(df, output_path)


# ============================================================
# TWO-SHEET EXCEL OUTPUT
# ============================================================

def save_two_sheet_workbook(df, output_path):
    """
    Save the final/checkpoint workbook with exactly two sheets:
      1. Image_Wise
      2. Summary
    """

    scores = pd.to_numeric(
        df["GT_similarity_score"],
        errors="coerce"
    ).dropna()

    total = len(df)
    scored = len(scores)
    blank_skipped = int((df.get("GT_status", pd.Series(index=df.index)) == "SKIPPED_BLANK").sum())
    gt_not_found = int((df.get("GT_status", pd.Series(index=df.index)) == "GT_NOT_FOUND").sum())

    if scored:
        mean_score = scores.mean()
        median_score = scores.median()
        min_score = scores.min()
        max_score = scores.max()
        std_score = scores.std(ddof=1) if scored > 1 else 0.0
    else:
        mean_score = median_score = min_score = max_score = std_score = None

    # Score-band counts
    band_order = [
        "0.90-1.00",
        "0.80-0.89",
        "0.70-0.79",
        "0.50-0.69",
        "0.30-0.49",
        "0.10-0.29",
        "0.00-0.09",
    ]

    band_counts = (
        df["GT_similarity_band"]
        .value_counts(dropna=True)
        .reindex(band_order, fill_value=0)
    )

    summary_rows = [
        ["Metric", "Value"],
        ["Total Images", total],
        ["Successfully Scored", scored],
        ["Blank Responses Skipped", blank_skipped],
        ["GT Not Found", gt_not_found],
        ["Unscored", total - scored],
        ["Average Similarity", mean_score],
        ["Median Similarity", median_score],
        ["Minimum Similarity", min_score],
        ["Maximum Similarity", max_score],
        ["Standard Deviation", std_score],
        ["", ""],
        ["Score Band", "Number of Images"],
    ]

    for band in band_order:
        summary_rows.append([
            band,
            int(band_counts.get(band, 0))
        ])

    summary_df = pd.DataFrame(
        summary_rows[1:],
        columns=summary_rows[0]
    )

    output_path = Path(output_path)
    tmp_path = Path(str(output_path) + ".tmp.xlsx")

    with pd.ExcelWriter(
        tmp_path,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Image_Wise",
            index=False
        )

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        # Basic formatting
        workbook = writer.book

        image_sheet = workbook["Image_Wise"]
        summary_sheet = workbook["Summary"]

        image_sheet.freeze_panes = "A2"
        summary_sheet.freeze_panes = "A2"

        # Auto-size columns
        for sheet in [image_sheet, summary_sheet]:
            for column_cells in sheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))

                sheet.column_dimensions[column_letter].width = min(
                    max(max_length + 2, 12),
                    60
                )

    os.replace(tmp_path, output_path)



# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Standalone semantic judge: ablation response vs GT"
    )

    parser.add_argument(
        "--responses",
        required=True,
        help="Ablation response XLSX"
    )

    parser.add_argument(
        "--gt1",
        required=True,
        help="CLAUDE.xlsx"
    )

    parser.add_argument(
        "--gt2",
        required=False,
        default=None,
        help="claude-Supplementary_Data_1.xlsx (optional; omit if all GT is in --gt1)"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model"
    )

    parser.add_argument(
        "--output",
        default="ablation_GT_judged.xlsx",
        help="Output XLSX"
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per failed judge call"
    )

    args = parser.parse_args()

    if args.output is None:
        response_stem = Path(args.responses).stem
        safe_stem = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            response_stem
        )
        args.output = f"{safe_stem}_GT_judged.xlsx"

    print("=" * 70)
    print("STANDALONE ABLATION: AI RESPONSE vs GROUND TRUTH")
    print("=" * 70)

    print(f"Responses : {args.responses}")
    print(f"GT 1     : {args.gt1}")
    print(f"GT 2     : {args.gt2 if args.gt2 else '(not provided)'}")
    print(f"Model    : {args.model}")
    print(f"Output   : {args.output}")

    print("=" * 70)

    # --------------------------------------------------------
    # Load response file
    # --------------------------------------------------------

    df = pd.read_excel(
        args.responses,
        header=0
    )

    if "Image_id" not in df.columns:
        raise ValueError(
            "Response file must contain 'Image_id'. "
            f"Found: {list(df.columns)}"
        )

    if "response" not in df.columns:
        raise ValueError(
            "Response file must contain 'response'. "
            f"Found: {list(df.columns)}"
        )

    print(f"Responses: {len(df)}")

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    gt_map = load_ground_truths(
        args.gt1,
        args.gt2 if args.gt2 else None
    )

    print(f"GT entries: {len(gt_map)}")

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    output_path = Path(args.output)

    if output_path.exists():

        print()
        print(f"Existing checkpoint found: {output_path}")

        old = pd.read_excel(
            output_path
        )

        required_checkpoint_columns = {
            "GT_similarity_score",
            "GT_similarity_band",
            "GT_similarity_reason"
        }

        if (
            len(old) == len(df)
            and required_checkpoint_columns.issubset(old.columns)
        ):

            df = old

            print(
                "Checkpoint loaded. "
                "Already-scored rows will be skipped."
            )

        else:

            print(
                "Existing checkpoint does not match this input. "
                "Starting fresh."
            )

    # --------------------------------------------------------
    # Output columns
    # --------------------------------------------------------

    output_columns = [
        "GROUND TRUTH",
        "GT_similarity_score",
        "GT_similarity_band",
        "GT_similarity_reason",
        "GT_source",
        "GT_judge_capped",
        "GT_status"
    ]

    for col in output_columns:

        if col not in df.columns:
            df[col] = None

    # --------------------------------------------------------
    # Process rows
    # --------------------------------------------------------

    newly_processed = 0

    for idx in range(len(df)):

        # ----------------------------------------------------
        # Resume:
        # if score already exists, do not call Ollama again.
        # ----------------------------------------------------

        existing_score = df.at[
            idx,
            "GT_similarity_score"
        ]

        existing_status = df.at[idx, "GT_status"] if "GT_status" in df.columns else None

        if pd.notna(existing_score):

            print(
                f"[{idx + 1}/{len(df)}] "
                f"{df.at[idx, 'Image_id']} "
                f"-> already judged: {existing_score}",
                flush=True
            )

            continue

        if str(existing_status).strip() == "SKIPPED_BLANK":

            print(
                f"[{idx + 1}/{len(df)}] "
                f"{df.at[idx, 'Image_id']} "
                f"-> already skipped (blank response)",
                flush=True
            )

            continue

        image_id = df.at[
            idx,
            "Image_id"
        ]

        print(
            f"[{idx + 1}/{len(df)}] {image_id}",
            flush=True
        )

        # ----------------------------------------------------
        # Find matching GT
        # ----------------------------------------------------

        gt_info = find_gt(
            image_id,
            gt_map
        )

        if gt_info is None:

            print(
                "  GT NOT FOUND",
                flush=True
            )

            df.at[
                idx,
                "GT_similarity_reason"
            ] = "Ground truth not found for this image ID."

            df.at[
                idx,
                "GT_judge_capped"
            ] = False

            df.at[
                idx,
                "GT_status"
            ] = "GT_NOT_FOUND"

        else:

            ground_truth = gt_info["ground_truth"]
            source = gt_info["source"]

            response = df.at[
                idx,
                "response"
            ]

            # Store GT immediately.
            df.at[
                idx,
                "GROUND TRUTH"
            ] = ground_truth

            df.at[
                idx,
                "GT_source"
            ] = source

            df.at[
                idx,
                "GT_judge_capped"
            ] = False

            # ------------------------------------------------
            # Empty response
            # ------------------------------------------------

            if pd.isna(response) or not str(response).strip():

                # Blank responses are skipped completely. They are NOT sent
                # to Ollama, receive NO score, and are excluded from all
                # similarity statistics.
                df.at[idx, "GT_status"] = "SKIPPED_BLANK"
                df.at[idx, "GT_similarity_reason"] = "Blank AI response skipped; no similarity score assigned."

                newly_processed += 1
                save_checkpoint(df, output_path)

                print(
                    "  BLANK RESPONSE -> SKIPPED (no score)",
                    flush=True
                )
                print(
                    f"  CHECKPOINT SAVED -> {output_path}",
                    flush=True
                )
                continue

            else:

                # ------------------------------------------------
                # Semantic judge
                # ------------------------------------------------

                score, band, reason, _ = judge_pair(
                    gt=ground_truth,
                    response=response,
                    model=args.model,
                    retries=args.retries
                )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            df.at[
                idx,
                "GT_similarity_score"
            ] = score

            df.at[
                idx,
                "GT_similarity_band"
            ] = band

            df.at[
                idx,
                "GT_similarity_reason"
            ] = reason

            df.at[
                idx,
                "GT_status"
            ] = "SCORED"

            print(
                f"  score={score:.2f} | "
                f"{band} | {reason}",
                flush=True
            )

        newly_processed += 1

        # ----------------------------------------------------
        # SAVE AFTER EVERY ROW
        # ----------------------------------------------------

        save_checkpoint(
            df,
            output_path
        )

        print(
            f"  CHECKPOINT SAVED -> {output_path}",
            flush=True
        )

    # --------------------------------------------------------
    # Final two-sheet workbook
    # --------------------------------------------------------

    save_two_sheet_workbook(df, output_path)

    scores = pd.to_numeric(
        df["GT_similarity_score"],
        errors="coerce"
    ).dropna()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Total rows       : {len(df)}")
    print(f"Scored rows      : {len(scores)}")
    print(f"Blank skipped    : {int((df.get('GT_status', pd.Series(index=df.index)) == 'SKIPPED_BLANK').sum())}")
    print(f"GT not found     : {int((df.get('GT_status', pd.Series(index=df.index)) == 'GT_NOT_FOUND').sum())}")
    print(
        f"Unscored rows    : "
        f"{len(df) - len(scores)}"
    )

    if len(scores):

        print(
            f"Mean score       : {scores.mean():.4f}"
        )

        print(
            f"Median score     : {scores.median():.4f}"
        )

        print(
            f"Minimum score    : {scores.min():.2f}"
        )

        print(
            f"Maximum score    : {scores.max():.2f}"
        )

    print()
    print(
        f"Final file saved : {output_path.resolve()}"
    )

    # --------------------------------------------------------
    # Direct Colab download
    # --------------------------------------------------------

    try:

        from google.colab import files

        print()
        print("Starting Colab download...")

        files.download(
            str(output_path)
        )

    except ImportError:
        pass

    except Exception as e:
        print(
            f"Could not start Colab download: {e}"
        )


if __name__ == "__main__":
    main()
